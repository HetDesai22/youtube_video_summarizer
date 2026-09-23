import time
from contextlib import contextmanager

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ..services.audio import extract_audio
from ..services.embeddings import preload_model_async as preload_embeddings_async
from ..services.question_detection import detect_and_save_questions_async, serialize_questions
from ..services.summarization import preload_model_async as preload_mistral_async
from ..services.summarization import summarize_transcript
from ..services.transcript_processing import process_transcript
from ..services.transcription import preload_model_async as preload_whisper_async
from ..services.transcription import transcribe_audio
from ..services.transcription import unload_model as unload_whisper
from ..services.validators.youtube_url import validate_youtube_url
from ..services.video_index import (
    get_cached_questions,
    get_cached_summary,
    index_video,
    save_summary,
)
from ..services.youtube import check_video_accessibility


class _StageTimer:
    """Records wall-clock seconds per pipeline stage (returned in the response
    so slow stages can be identified without reading server logs)."""

    def __init__(self):
        self.seconds = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - start, 1)

    def report(self):
        return {**self.seconds, "total": round(time.perf_counter() - self._started, 1)}


@require_GET
def check_video(request):
    url = request.GET.get("url", "")
    timer = _StageTimer()

    validation = validate_youtube_url(url)
    if not validation["valid"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "url_validation",
                "message": validation["error"],
            }
        )

    # A video that was already processed is served instantly from the database
    # (no download, transcription or summarization). ?refresh=1 forces a re-run.
    if request.GET.get("refresh") != "1":
        cached = get_cached_summary(validation["video_id"])
        if cached is not None:
            cached_questions = get_cached_questions(cached) or []
            return JsonResponse(
                {
                    "success": True,
                    "message": "Loaded from a previous summary.",
                    "data": {
                        "cached": True,
                        "timings": timer.report(),
                        "video_id": cached.video_id,
                        "chat_available": True,
                        "title": cached.title,
                        "channel": cached.channel,
                        "duration": cached.duration,
                        "thumbnail": cached.thumbnail,
                        "summary": cached.summary,
                        "questions": serialize_questions(cached.video_id, cached_questions),
                        "questions_pending": False,
                    },
                }
            )

    # Kick off both heavy models loading in the background now, so their
    # load time overlaps with the network-bound accessibility check and
    # audio download below instead of adding to the total request time
    # only once summarization is reached.
    preload_whisper_async()
    preload_mistral_async()
    preload_embeddings_async()

    with timer.stage("accessibility"):
        accessibility = check_video_accessibility(validation["url"])
    if not accessibility["accessible"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "accessibility",
                "message": (
                    "The YouTube video is unavailable or cannot be accessed. "
                    "Please check the URL or make sure the video is publicly available."
                ),
            }
        )

    with timer.stage("audio_extraction"):
        audio = extract_audio(validation["url"], validation["video_id"])
    if not audio["success"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "audio_extraction",
                "message": audio["error"],
            }
        )

    with timer.stage("transcription"):
        transcription = transcribe_audio(audio["path"])
    # Whisper is only needed for this step; free its VRAM for Mistral.
    unload_whisper()
    if not transcription["success"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "transcription",
                "message": transcription["error"],
            }
        )

    with timer.stage("transcript_processing"):
        processed = process_transcript(transcription)
    if not processed["success"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "transcript_processing",
                "message": processed["error"],
            }
        )

    with timer.stage("indexing"):
        indexed = index_video(
            validation["video_id"],
            validation["url"],
            {
                "title": accessibility["title"],
                "channel": accessibility["channel"],
                "duration": accessibility["duration"],
                "thumbnail": accessibility["thumbnail"],
            },
            processed,
        )

    with timer.stage("summarization"):
        summarization = summarize_transcript(processed)
    if not summarization["success"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "summarization",
                "message": summarization["error"],
            }
        )

    # Only cache a well-formed structured summary; a degraded plain-text
    # fallback ("raw_output" present) is not worth serving repeatedly.
    if indexed["success"] and "raw_output" not in summarization:
        save_summary(
            indexed["video"],
            {
                key: summarization.get(key)
                for key in ("summary", "main_topics", "key_points", "conclusions", "raw_output")
            },
        )

    # Question detection is moved off the critical path: it costs ~35-40s of
    # GPU time that summarization doesn't need, so it runs in a background
    # thread and this response doesn't wait for it. Dispatched only NOW -
    # after summarization has already finished and released the shared GPU
    # lock - so it can never win a race for that lock ahead of summarization
    # and end up delaying the very response it's supposed to not block. The
    # frontend polls this same endpoint again shortly after to pick up the
    # questions once they're ready - see detect_and_save_questions_async's
    # docstring.
    if indexed["success"]:
        detect_and_save_questions_async(indexed["video"], processed)

    return JsonResponse(
        {
            "success": True,
            "message": "Summarization completed.",
            "data": {
                "cached": False,
                "timings": {**timer.report(), "summarization_detail": summarization.get("timings")},
                "video_id": validation["video_id"],
                "chat_available": indexed["success"],
                "questions": [],
                "questions_pending": indexed["success"],
                "title": accessibility["title"],
                "channel": accessibility["channel"],
                "duration": accessibility["duration"],
                "thumbnail": accessibility["thumbnail"],
                "audio": {
                    "format": audio["format"].upper(),
                    "sample_rate": f"{audio['sample_rate'] // 1000} kHz",
                    "channels": "Mono" if audio["channels"] == 1 else str(audio["channels"]),
                },
                "transcript": {
                    "language": processed["language"],
                    "duration": processed["duration"],
                    "segments": processed["segments"],
                    "full_text": processed["full_text"],
                    "chunks": processed["chunks"],
                    "device": transcription["device"],
                    "model": transcription["model"],
                },
                "summary": {
                    "summary": summarization["summary"],
                    "main_topics": summarization.get("main_topics"),
                    "key_points": summarization.get("key_points"),
                    "conclusions": summarization.get("conclusions"),
                    "raw_output": summarization.get("raw_output"),
                    "chunk_summaries": summarization["chunk_summaries"],
                    "model": summarization["model"],
                    "device": summarization["device"],
                },
            },
        }
    )

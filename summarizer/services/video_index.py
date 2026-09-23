import logging

from django.db import transaction

from ..models import TranscriptChunk, Transcript, Video, VideoQuestion
from . import embeddings

logger = logging.getLogger(__name__)

_INDEX_ERROR = "We couldn't prepare this video for chat. Please try again."


def get_indexed_video(video_id):
    """Return the Video if it has been fully indexed (READY with chunks)."""
    video = Video.objects.filter(video_id=video_id, status=Video.Status.READY).first()
    if video is not None and video.chunks.exists():
        return video
    return None


# Bump whenever the summary's style/length requirements change: cached summaries
# from an older version are then ignored and regenerated instead of being served
# in the old style. (v3: summary is a list of 15+ lines, joined by newlines.)
SUMMARY_VERSION = 3


def get_cached_summary(video_id):
    """Return the Video if it was already fully processed AND summarised in the
    current summary style (ready to serve instantly, and to chat with),
    otherwise None.
    """
    video = Video.objects.filter(
        video_id=video_id,
        status=Video.Status.READY,
        summary__version=SUMMARY_VERSION,
    ).first()
    if video is not None and video.chunks.exists():
        return video
    return None


def save_summary(video, summary):
    """Persist the final summary payload on the video so it can be reused."""
    Video.objects.filter(pk=video.pk).update(summary={**summary, "version": SUMMARY_VERSION})


def get_cached_questions(video):
    """Return previously-detected questions for `video` (chronological), or
    None if question detection hasn't run for it yet.
    """
    if not video.questions.exists():
        return None
    return list(video.questions.order_by("start"))


def save_questions(video, questions):
    """Persist detected questions for `video`, replacing any previous set
    (so re-running detection never appends duplicates alongside stale ones).
    """
    VideoQuestion.objects.filter(video=video).delete()
    VideoQuestion.objects.bulk_create(
        [
            VideoQuestion(
                video=video,
                question=question["question"],
                start=question["start"],
                end=question["end"],
                source_segment_index=question.get("segment_index"),
            )
            for question in questions
        ]
    )


def index_video(video_id, url, metadata, processed):
    """Persist a processed transcript and its chunk embeddings so the video
    can be chatted with.

    `processed` is the dict returned by transcript_processing.process_transcript
    (language, duration, full_text, chunks[...]). If the video has already been
    indexed, the stored embeddings are reused and nothing is re-embedded.

    Returns {"success": True, "video": Video, "reused": bool} or
    {"success": False, "error": str}. Never raises.
    """
    metadata = metadata or {}

    existing = get_indexed_video(video_id)
    if existing is not None:
        logger.info("[VIDEO CHAT] Video %s already indexed; reusing embeddings", video_id)
        return {"success": True, "video": existing, "reused": True}

    chunks = processed.get("chunks") or []
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        logger.error("[VIDEO CHAT] No chunks to index for video %s", video_id)
        return {"success": False, "error": _INDEX_ERROR}

    video, _ = Video.objects.update_or_create(
        video_id=video_id,
        defaults={
            "url": url,
            "title": metadata.get("title", "") or "",
            "channel": metadata.get("channel", "") or "",
            "duration": metadata.get("duration", "") or "",
            "thumbnail": metadata.get("thumbnail", "") or "",
            "status": Video.Status.PROCESSING,
        },
    )

    try:
        logger.info("[VIDEO CHAT] Embedding %d chunks for video %s", len(texts), video_id)
        vectors = embeddings.embed_texts(texts)

        with transaction.atomic():
            Transcript.objects.update_or_create(
                video=video,
                defaults={
                    "language": processed.get("language") or "",
                    "duration_seconds": processed.get("duration"),
                    "full_text": processed.get("full_text", ""),
                },
            )
            video.chunks.all().delete()
            TranscriptChunk.objects.bulk_create(
                [
                    TranscriptChunk(
                        video=video,
                        chunk_id=chunk["chunk_id"],
                        start=chunk["start"],
                        end=chunk["end"],
                        text=chunk["text"],
                        word_count=chunk["word_count"],
                        embedding=vector,
                    )
                    for chunk, vector in zip(chunks, vectors)
                ]
            )
            video.status = Video.Status.READY
            video.save(update_fields=["status", "updated_at"])
    except Exception:
        logger.exception("[VIDEO CHAT] Failed to index video %s", video_id)
        Video.objects.filter(pk=video.pk).update(status=Video.Status.FAILED)
        return {"success": False, "error": _INDEX_ERROR}

    return {"success": True, "video": video, "reused": False}

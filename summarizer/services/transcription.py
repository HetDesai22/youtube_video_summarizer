import gc
import logging
import threading
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Greedy decoding (1) instead of beam search (5): measured on a real 10-minute
# slice, 2.5x faster (25.5 s vs 64.8 s) with 96% word-level agreement to the
# beam-5 transcript. Raise it back to 5 if maximum accuracy matters more than
# speed. (int8_float16 is not supported by this GPU's cuBLAS, so not an option.)
BEAM_SIZE = 1
TEMPERATURE = 0.0

# Disabled by default: Silero VAD (faster-whisper's vad_filter) misclassifies
# music/singing as non-speech at default sensitivity. Verified against a real
# YouTube music video, which VAD reduced from 28 correct segments to 0. Since
# a large share of real YouTube content has background music, defaulting to
# vad_filter=False is safer for "reliable transcription of normal YouTube
# speech" than defaulting it on. Re-enable (with tuned parameters) once this
# has been validated across more content types.
VAD_FILTER = False

_MISSING_FILE_ERROR = "Audio file could not be found."
_GENERIC_ERROR = "We couldn't transcribe the audio. Please try again."

_model = None
_model_lock = threading.Lock()


def _load_model():
    """Lazily load and cache the faster-whisper model for reuse across
    requests within this process. Raises on failure; callers decide how to
    present that to the user. No automatic CPU fallback: a GPU failure is
    reported as a failure, not silently downgraded.
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info(
                "Loading faster-whisper model '%s' on device=%s compute_type=%s",
                MODEL_SIZE, DEVICE, COMPUTE_TYPE,
            )
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    return _model


def unload_model():
    """Free the Whisper model (and its GPU memory). On an 8 GB GPU the model
    would otherwise stay resident next to 4-bit Mistral and starve chat /
    summarization of memory. It is reloaded lazily for the next video.
    """
    global _model

    with _model_lock:
        if _model is None:
            return
        _model = None

    gc.collect()
    logger.info("Unloaded faster-whisper model to free GPU memory")


def preload_model_async():
    """Kick off model loading in a background thread if it isn't already
    loaded (or being loaded). Non-blocking - lets the model finish loading
    while other independent work (e.g. downloading audio) is in progress,
    instead of paying the full load cost only when transcription is first
    requested. Safe to call multiple times; a later call to `_load_model()`
    simply waits on the same lock rather than loading twice.
    """
    if _model is not None:
        return

    def _worker():
        try:
            _load_model()
        except Exception:
            logger.exception(
                "Background preload of faster-whisper failed; will retry on first real request"
            )

    threading.Thread(target=_worker, daemon=True).start()


def transcribe_audio(audio_path):
    """Transcribe a WAV file produced by the audio-extraction module using
    faster-whisper, preserving timestamped segments and detected language.

    Returns on success:
    {
        "success": True,
        "language": "en",
        "duration": 213.04,
        "segments": [{"start": 0.0, "end": 5.2, "text": "..."}, ...],
        "text": "...",
        "device": "cuda",
        "model": "small",
    }
    or {"success": False, "error": str} on failure. Technical exceptions
    (including CUDA/CTranslate2 errors) are logged, never returned to the
    caller.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.error("Audio file not found for transcription: %s", audio_path)
        return {"success": False, "error": _MISSING_FILE_ERROR}

    try:
        model = _load_model()
    except Exception:
        logger.exception(
            "Failed to initialize faster-whisper on device=%s compute_type=%s",
            DEVICE, COMPUTE_TYPE,
        )
        return {"success": False, "error": _GENERIC_ERROR}

    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            temperature=TEMPERATURE,
        )
        segments = [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in segments_iter
        ]
    except Exception:
        logger.exception("faster-whisper failed to transcribe %s", audio_path)
        return {"success": False, "error": _GENERIC_ERROR}

    full_text = " ".join(segment["text"] for segment in segments).strip()

    return {
        "success": True,
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
        "text": full_text,
        "device": DEVICE,
        "model": MODEL_SIZE,
    }

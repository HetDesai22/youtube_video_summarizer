import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_WORDS = 500

_WHITESPACE_RE = re.compile(r"\s+")

_MALFORMED_ERROR = "Transcript processing failed."
_NO_USABLE_SEGMENTS_ERROR = "No usable transcript was found."


@dataclass
class Segment:
    start: float
    end: float
    text: str

    def as_dict(self):
        return {"start": self.start, "end": self.end, "text": self.text}


def clean_text(text):
    """Conservatively clean transcript text.

    Only collapses whitespace and trims the ends. Never rewrites, paraphrases,
    or drops words - what the speaker said must remain unchanged.
    """
    if not text or not isinstance(text, str):
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_valid_raw_segment(segment):
    if not isinstance(segment, dict):
        return False

    start = segment.get("start")
    end = segment.get("end")
    text = segment.get("text")

    if isinstance(start, bool) or isinstance(end, bool):
        return False
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    if not math.isfinite(start) or not math.isfinite(end):
        return False
    if start < 0 or end < 0 or end < start:
        return False
    if not isinstance(text, str):
        return False

    return True


def validate_and_clean_segments(raw_segments):
    """Validate and clean raw faster-whisper segments.

    Segments with missing/invalid timestamps, missing text, or text that is
    empty/whitespace-only are dropped. A single malformed segment is logged
    and skipped rather than raising, so it can never take down the whole
    request.
    """
    cleaned = []
    for raw in raw_segments or []:
        try:
            if not _is_valid_raw_segment(raw):
                logger.warning("Skipping invalid transcript segment: %r", raw)
                continue

            text = clean_text(raw["text"])
            if not text:
                continue

            cleaned.append(
                Segment(start=float(raw["start"]), end=float(raw["end"]), text=text)
            )
        except Exception:
            logger.exception("Unexpected error validating transcript segment: %r", raw)
            continue

    return cleaned


def build_full_text(segments):
    return " ".join(segment.text for segment in segments).strip()


def chunk_transcript(segments, max_words=DEFAULT_CHUNK_WORDS):
    """Group cleaned segments into chunks of at most `max_words` words each.

    Segments are packed greedily and kept whole wherever possible, so chunk
    boundaries fall on segment boundaries rather than mid-sentence. If a
    single segment's own word count exceeds max_words, it is split at word
    boundaries (never mid-word) into multiple pieces, with its time range
    divided proportionally by word position so timing stays as accurate as
    possible without discarding any of its text.
    """
    chunks = []
    current_words = []
    current_start = None
    current_end = None

    def flush():
        if not current_words:
            return
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_words),
                "word_count": len(current_words),
            }
        )

    for segment in segments:
        words = segment.text.split()
        if not words:
            continue

        if len(words) > max_words:
            flush()
            current_words = []
            current_start = None
            current_end = None

            total = len(words)
            span = segment.end - segment.start
            for i in range(0, total, max_words):
                piece_words = words[i : i + max_words]
                piece_start = segment.start + span * (i / total)
                piece_end = segment.start + span * (min(i + len(piece_words), total) / total)
                chunks.append(
                    {
                        "chunk_id": len(chunks) + 1,
                        "start": piece_start,
                        "end": piece_end,
                        "text": " ".join(piece_words),
                        "word_count": len(piece_words),
                    }
                )
            continue

        if current_words and len(current_words) + len(words) > max_words:
            flush()
            current_words = []
            current_start = None
            current_end = None

        if current_start is None:
            current_start = segment.start
        current_end = segment.end
        current_words.extend(words)

    flush()
    return chunks


def process_transcript(transcription_result, max_chunk_words=DEFAULT_CHUNK_WORDS):
    """Turn a raw faster-whisper transcription result into a cleaned,
    chunked, structured transcript ready for a future summarization module.

    Raw transcription -> validate segments -> clean text -> drop invalid or
    empty segments -> build full transcript -> create chunks.

    Returns on success:
    {
        "success": True,
        "language": "en",
        "duration": 125.4,
        "segments": [{"start": 0.0, "end": 5.4, "text": "..."}, ...],
        "full_text": "...",
        "chunks": [{"chunk_id": 1, "start": 0.0, "end": ..., "text": ...,
                     "word_count": ...}, ...],
    }
    or {"success": False, "error": str} if the input is malformed or no
    usable segments remain. Never raises; deterministic for a given input.
    """
    if not isinstance(transcription_result, dict) or "segments" not in transcription_result:
        logger.error(
            "process_transcript received a malformed transcription result: %r",
            transcription_result,
        )
        return {"success": False, "error": _MALFORMED_ERROR}

    try:
        cleaned_segments = validate_and_clean_segments(transcription_result.get("segments"))
    except Exception:
        logger.exception("Unexpected error while processing transcript segments")
        return {"success": False, "error": _MALFORMED_ERROR}

    if not cleaned_segments:
        return {"success": False, "error": _NO_USABLE_SEGMENTS_ERROR}

    chunks = chunk_transcript(cleaned_segments, max_words=max_chunk_words)

    return {
        "success": True,
        "language": transcription_result.get("language"),
        "duration": transcription_result.get("duration"),
        "segments": [segment.as_dict() for segment in cleaned_segments],
        "full_text": build_full_text(cleaned_segments),
        "chunks": chunks,
    }

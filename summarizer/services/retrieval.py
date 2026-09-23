import logging

from pgvector.django import CosineDistance

from ..models import TranscriptChunk
from . import embeddings

logger = logging.getLogger(__name__)

# Configurable - tune during testing. Measured on a real 13-chunk video: questions
# the video answers scored 0.34-0.61 (terse technical ones ~0.20, since 500-word
# chunks dilute short queries); questions it does not cover scored <= 0.19.
# 0.25 sits on the safe side of that gap. A threshold reduces weak-match answers
# but does NOT guarantee factuality.
# TOP_K is 3, not 5: on an 8 GB GPU shared with 4-bit Mistral, five ~500-word
# chunks (~3.3K tokens) ran out of memory on a real video.
TOP_K = 3
SIMILARITY_THRESHOLD = 0.25


def retrieve_chunks(video, question, top_k=TOP_K):
    """Return the `top_k` transcript chunks of `video` most similar to
    `question`, best first. Search is always restricted to `video`, so chunks
    from other videos can never be returned.

    Each result: {"chunk_id", "text", "start", "end", "similarity"}.
    Raises if embedding fails; the chat service turns that into a clean error.
    """
    logger.info("[VIDEO CHAT] Generating question embedding")
    question_vector = embeddings.embed_text(question)

    logger.info("[VIDEO CHAT] Searching transcript chunks")
    rows = (
        TranscriptChunk.objects.filter(video=video)
        .annotate(distance=CosineDistance("embedding", question_vector))
        .order_by("distance")[:top_k]
    )

    results = [
        {
            "chunk_id": row.chunk_id,
            "text": row.text,
            "start": row.start,
            "end": row.end,
            "similarity": round(1.0 - row.distance, 4),
        }
        for row in rows
    ]
    logger.info("[VIDEO CHAT] Retrieved %d chunks", len(results))
    return results

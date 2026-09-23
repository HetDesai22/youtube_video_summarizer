import logging
import threading

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# CPU by default: MiniLM is tiny and fast enough on CPU, and it keeps the
# 8 GB GPU free for Whisper + 4-bit Mistral. Set to "cuda" to use the GPU.
DEVICE = "cpu"

BATCH_SIZE = 32

_model = None
_model_lock = threading.Lock()


def _load_model():
    """Lazily load and cache the embedding model - one instance per process,
    shared by chunk indexing and question embedding (they MUST use the same
    model for vector similarity to mean anything).
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info("[EMBEDDINGS] Loading %s on %s", MODEL_NAME, DEVICE)
            _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            logger.info("[EMBEDDINGS] Model loaded")

    return _model


def preload_model_async():
    """Load the embedding model in a background thread; no-op if loaded."""
    if _model is not None:
        return

    def _worker():
        try:
            _load_model()
        except Exception:
            logger.exception("[EMBEDDINGS] Background preload failed; will retry on first use")

    threading.Thread(target=_worker, daemon=True).start()


def get_embedding_dimension():
    return _load_model().get_sentence_embedding_dimension()


def embed_texts(texts):
    """Embed a list of strings; returns a list of L2-normalised float vectors
    (so cosine similarity == dot product). Raises on model failure - callers
    (indexing / retrieval) translate that into a clean user-facing error.
    """
    if not texts:
        return []

    model = _load_model()
    vectors = model.encode(
        list(texts),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_text(text):
    return embed_texts([text])[0]

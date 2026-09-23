"""Tests 2 & 3: store chunks + embeddings in PostgreSQL/pgvector, then retrieve.

Uses two throwaway videos (removed at the end) to also verify that retrieval is
isolated per video.

    set HF_HOME=E:\\hf_cache
    venv\\Scripts\\python.exe scripts\\test_retrieval.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "videosummary.settings")

import django  # noqa: E402

django.setup()

from summarizer.models import TranscriptChunk, Video  # noqa: E402
from summarizer.services.retrieval import retrieve_chunks  # noqa: E402
from summarizer.services.video_index import index_video  # noqa: E402

A, B = "TESTVIDEO_A", "TESTVIDEO_B"


def make_processed(texts):
    chunks = [
        {"chunk_id": i, "start": i * 60.0, "end": i * 60.0 + 55.0, "text": t, "word_count": len(t.split())}
        for i, t in enumerate(texts, start=1)
    ]
    return {"language": "en", "duration": len(texts) * 60.0, "full_text": " ".join(texts), "chunks": chunks}


video_a_texts = [
    "Electric vehicles use large lithium-ion battery packs to store electrical energy, "
    "which then powers the electric motor that drives the wheels.",
    "To make fresh pasta you mix flour and eggs, knead the dough for ten minutes, "
    "and then let it rest before rolling it out.",
    "The football match ended in a draw after both teams scored in the final minutes.",
]
video_b_texts = [
    "The internal combustion engine burns petrol in its cylinders and the pistons drive the crankshaft.",
]


def main():
    try:
        print("=== TEST 2: store chunks + embeddings ===")
        ra = index_video(A, "https://youtu.be/TESTVIDEO_A", {"title": "EV talk"}, make_processed(video_a_texts))
        rb = index_video(B, "https://youtu.be/TESTVIDEO_B", {"title": "Engines"}, make_processed(video_b_texts))
        print("index A:", ra["success"], "| index B:", rb["success"])
        video_a = Video.objects.get(video_id=A)
        print("Video A status:", video_a.status, "| chunks in DB:", TranscriptChunk.objects.filter(video=video_a).count())
        sample = TranscriptChunk.objects.filter(video=video_a).first()
        print("stored embedding dims:", len(sample.embedding), "| start/end:", sample.start, sample.end)

        again = index_video(A, "https://youtu.be/TESTVIDEO_A", {}, make_processed(video_a_texts))
        print("re-index reuses embeddings:", again["reused"])

        print("\n=== TEST 3: vector retrieval ===")
        for r in retrieve_chunks(video_a, "How do electric vehicles store energy?", top_k=3):
            print(f"  chunk {r['chunk_id']}  sim={r['similarity']}  {r['text'][:70]}")

        print("\n=== Isolation: ask Video A about engines ===")
        results = retrieve_chunks(video_a, "What did the speaker say about combustion engines?", top_k=5)
        b_leaked = any("combustion" in r["text"] for r in results)
        print("  returned chunk ids:", [r["chunk_id"] for r in results], "| Video B text leaked:", b_leaked)
        assert not b_leaked, "Video B chunk leaked into Video A retrieval!"
        assert len(results) == 3, "Only Video A's 3 chunks should be searchable"
    finally:
        Video.objects.filter(video_id__in=[A, B]).delete()
        print("\n(test videos cleaned up)")


if __name__ == "__main__":
    main()

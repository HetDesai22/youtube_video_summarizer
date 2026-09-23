"""Tests 4, 7, 8: grounded Mistral answers, conversation memory, out-of-context.

Indexes a small synthetic video (removed at the end) and chats with it.

    set HF_HOME=E:\\hf_cache
    venv\\Scripts\\python.exe scripts\\test_video_chat.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "videosummary.settings")

import django  # noqa: E402

django.setup()

from summarizer.models import Video  # noqa: E402
from summarizer.services.video_chat import chat_with_video, get_or_create_session  # noqa: E402
from summarizer.services.video_index import index_video  # noqa: E402

VID = "TESTVIDEO_CHAT"

TEXTS = [
    "Welcome to this video about electric vehicles. Today we cover how batteries work, "
    "how charging works, and why electric cars are becoming popular.",
    "Electric vehicles store energy in large lithium-ion battery packs. The pack is made of "
    "thousands of small cells, and it powers the electric motor that turns the wheels. "
    "A typical pack holds between sixty and one hundred kilowatt hours.",
    "Charging at home takes around eight hours overnight, while a fast charger can add "
    "eighty percent of the range in about thirty minutes.",
    "Regenerative braking recovers energy when the driver slows down and sends it back into the battery, "
    "which extends the driving range by roughly ten percent.",
]


def ask(session, question):
    t0 = time.time()
    result = chat_with_video(session, question)
    print(f"\nQ: {question}   ({time.time() - t0:.1f}s)")
    if not result["success"]:
        print("  ERROR:", result["error"])
        return result
    print("A:", result["answer"])
    for s in result["sources"]:
        print(f"   source: {s['label']}  sim={s['similarity']}  {s['youtube_url']}")
    if not result["sources"]:
        print("   (no sources)")
    return result


def main():
    chunks = [
        {"chunk_id": i, "start": i * 60.0, "end": i * 60.0 + 58.0, "text": t, "word_count": len(t.split())}
        for i, t in enumerate(TEXTS, start=1)
    ]
    processed = {"language": "en", "duration": 300.0, "full_text": " ".join(TEXTS), "chunks": chunks}
    try:
        r = index_video(VID, f"https://www.youtube.com/watch?v={VID}", {"title": "EV basics"}, processed)
        assert r["success"], r
        video = Video.objects.get(video_id=VID)
        session = get_or_create_session(video)

        print("=== TEST 4: grounded answer ===")
        ask(session, "What does the video say about storing energy?")

        print("\n=== TEST 7: conversation ===")
        ask(session, "What is the video about?")
        ask(session, "What did they say about that?")
        ask(session, "How long does it take to charge?")
        ask(session, "And with the fast one?")

        print("\n=== TEST 8: out-of-context ===")
        ask(session, "What is the capital of France?")
        ask(session, "Who won the 2018 FIFA World Cup?")
    finally:
        Video.objects.filter(video_id=VID).delete()
        print("\n(test video cleaned up)")


if __name__ == "__main__":
    main()

"""End-to-end chat test over HTTP against the running dev server (Tests 5-8).

The video must already have been processed via /check-video/.

    venv\\Scripts\\python.exe scripts\\test_chat_http.py [VIDEO_ID] [BASE_URL]
"""
import json
import os
import sys
import time
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "videosummary.settings")

import django  # noqa: E402

django.setup()

from summarizer.models import TranscriptChunk  # noqa: E402

VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "2hP6ZSVibHg"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8001"

jar = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
opener.open(BASE + "/").read()  # sets the csrftoken cookie, like the browser does
csrf = next(c.value for c in jar if c.name == "csrftoken")

session_id = None
real_chunks = {c.chunk_id: (c.start, c.end) for c in TranscriptChunk.objects.filter(video__video_id=VIDEO_ID)}


def ask(question):
    global session_id
    body = json.dumps({"video_id": VIDEO_ID, "session_id": session_id, "message": question}).encode()
    req = urllib.request.Request(
        BASE + "/chat/", data=body, headers={"Content-Type": "application/json", "X-CSRFToken": csrf}
    )
    t0 = time.time()
    try:
        data = json.loads(opener.open(req, timeout=300).read())
    except urllib.error.HTTPError as exc:
        data = json.loads(exc.read())
    print(f"\nQ: {question}   ({time.time() - t0:.1f}s)")
    if not data.get("success"):
        print("  ERROR:", data.get("error"))
        return data
    session_id = data["session_id"]
    print("A:", data["answer"])
    for s in data["sources"]:
        ok = real_chunks.get(s["chunk_id"]) == (s["start"], s["end"])
        print(f"   source {s['label']}  sim={s['similarity']}  timestamps match DB chunk: {ok}")
        print(f"          {s['youtube_url']}")
        assert ok, "source timestamps do not match the stored chunk!"
    if not data["sources"]:
        print("   (no sources)")
    return data


print(f"video {VIDEO_ID}: {len(real_chunks)} chunks in PostgreSQL")
print("\n=== TEST 5: several questions ===")
ask("What is the video about?")
ask("How do I trim a video using ffmpeg?")
ask("How do I convert mp4 to mkv?")

print("\n=== TEST 7: conversation memory (new session) ===")
session_id = None
ask("What tool is used to install ffmpeg on a Mac?")
ask("How do I install it?")
ask("What about converting mp4 to mkv with it?")

print("\n=== TEST 8a: out-of-context INSIDE a conversation (must not be contaminated) ===")
ask("What is the capital of France?")
ask("Explain how neural networks are trained.")
print("\n--- and the conversation still works afterwards:")
ask("How do I trim a video using ffmpeg?")

print("\n=== TEST 8b: out-of-context in a brand-new session ===")
session_id = None
ask("Who won the 2018 FIFA World Cup?")
ask("What did the speaker say about electric cars?")

print("\n=== error handling ===")
req = urllib.request.Request(
    BASE + "/chat/",
    data=json.dumps({"video_id": "nonexistent1", "message": "hi"}).encode(),
    headers={"Content-Type": "application/json", "X-CSRFToken": csrf},
)
try:
    opener.open(req)
except urllib.error.HTTPError as exc:
    print("unknown video ->", exc.code, json.loads(exc.read()))

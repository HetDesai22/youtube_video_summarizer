"""Standalone Module 5 test script - run outside the Django dev server.

Progressively verifies Mistral summarization:
  1. GPU / model placement verification
  2. Arbitrary text summarization (sanity check)
  3. Single real transcript chunk
  4. Multiple real transcript chunks
  5. Full processed transcript -> summarize_transcript()

Run with:
    venv\\Scripts\\python.exe scripts\\test_mistral.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "videosummary.settings")

import django  # noqa: E402

django.setup()

import torch  # noqa: E402

from summarizer.services.summarization import summarize_text, summarize_transcript  # noqa: E402
from summarizer.services.transcript_processing import process_transcript  # noqa: E402
from summarizer.services.transcription import transcribe_audio  # noqa: E402


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def verify_gpu():
    section("STEP 0 - GPU verification")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is not available - Mistral will fail to load in 4-bit as configured.")


def test_arbitrary_text():
    section("STEP 1 - Arbitrary text summarization")
    text = (
        "Artificial intelligence is being used in modern vehicles to improve safety, "
        "navigation, predictive maintenance, and driver assistance. Machine learning "
        "models can analyze sensor data and identify potential problems before they "
        "become serious."
    )
    t0 = time.time()
    summary = summarize_text(text)
    elapsed = time.time() - t0

    from summarizer.services import summarization as summarization_module

    device = next(summarization_module._model.parameters()).device
    print("Model device:", device)
    print(f"Generated in {elapsed:.1f}s")
    print("\nInput:")
    print(" ", text)
    print("\nSummary:")
    print(" ", summary)
    return summary


def get_real_processed_transcript():
    section("Preparing a real transcript (reusing Module 2/3/4 output)")
    wav_path = Path("media/audio/dQw4w9WgXcQ.wav")
    if not wav_path.exists():
        print(f"No cached audio found at {wav_path}; transcribing will be skipped.")
        return None

    raw = transcribe_audio(str(wav_path))
    if not raw["success"]:
        print("transcribe_audio failed:", raw["error"])
        return None

    processed = process_transcript(raw, max_chunk_words=40)  # force multiple chunks
    if not processed["success"]:
        print("process_transcript failed:", processed["error"])
        return None

    print(f"Processed transcript ready: {len(processed['chunks'])} chunks")
    return processed


def test_single_chunk(processed):
    section("STEP 2 - Single real transcript chunk")
    if not processed or not processed["chunks"]:
        print("Skipped: no processed transcript available.")
        return
    chunk = processed["chunks"][0]
    print("Chunk text:", chunk["text"][:150] + ("..." if len(chunk["text"]) > 150 else ""))
    t0 = time.time()
    summary = summarize_text(chunk["text"])
    print(f"Generated in {time.time() - t0:.1f}s")
    print("\nChunk summary:")
    print(" ", summary)


def test_multiple_chunks(processed):
    section("STEP 3 - Multiple real transcript chunks")
    if not processed or len(processed["chunks"]) < 2:
        print("Skipped: fewer than 2 chunks available.")
        return
    for chunk in processed["chunks"][:3]:
        t0 = time.time()
        summary = summarize_text(chunk["text"])
        print(f"\nChunk {chunk['chunk_id']} ({time.time() - t0:.1f}s):")
        print(" ", summary)


def test_full_pipeline(processed):
    section("STEP 4 - Full processed transcript -> summarize_transcript()")
    if not processed:
        print("Skipped: no processed transcript available.")
        return
    t0 = time.time()
    result = summarize_transcript(processed)
    elapsed = time.time() - t0

    print(f"Completed in {elapsed:.1f}s")
    print("success:", result["success"])
    if not result["success"]:
        print("error:", result["error"])
        return

    print("model:", result["model"])
    print("device:", result["device"])
    print("num chunk summaries:", len(result["chunk_summaries"]))
    for cs in result["chunk_summaries"]:
        print(f"  chunk {cs['chunk_id']} [{cs['start']:.1f}-{cs['end']:.1f}]: {cs['summary'][:80]}...")

    print("\nFinal summary:")
    print(" ", result["summary"])
    if "main_topics" in result:
        print("\nMain topics:", result["main_topics"])
        print("Key points:", result["key_points"])
        print("Conclusions:", result["conclusions"])
    else:
        print("\n(Structured JSON parsing failed - raw_output returned instead)")
        print("raw_output:", result["raw_output"][:300])


if __name__ == "__main__":
    verify_gpu()
    test_arbitrary_text()
    processed = get_real_processed_transcript()
    test_single_chunk(processed)
    test_multiple_chunks(processed)
    test_full_pipeline(processed)

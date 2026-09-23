"""Standalone tests for automatic question detection (spec Tests 1-6), run
against the real Mistral model (no Django test client / mocking).

    set HF_HOME=E:\\hf_cache
    venv\\Scripts\\python.exe scripts\\test_question_detection.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "videosummary.settings")

import django  # noqa: E402

django.setup()

from summarizer.services.question_detection import detect_questions, serialize_questions  # noqa: E402


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def run(label, segments, expect_none=False):
    print(f"\n=== {label} ===")
    result = detect_questions({"segments": segments})
    print("success:", result["success"])
    if not result["success"]:
        print("error:", result["error"])
        return result
    questions = result["questions"]
    if expect_none:
        print("PASS" if not questions else f"FAIL - expected none, got {questions}")
    for q in serialize_questions("abcDEFghijk", questions):
        print(f"  [{q['timestamp']}] {q['question']}  ({q['youtube_url']})")
    return result


# Test 1: explicit question with literal '?'
run(
    "Test 1: explicit question",
    [
        seg(0.0, 4.0, "What is machine learning?"),
        seg(4.0, 12.0, "Machine learning is a method for learning patterns from data."),
    ],
)

# Test 2: plain statement, no question
run(
    "Test 2: no question",
    [seg(0.0, 6.0, "Machine learning is a method for learning patterns from data.")],
    expect_none=True,
)

# Test 3: spoken question without literal '?'
run(
    "Test 3: spoken question, no '?'",
    [
        seg(0.0, 5.0, "Now we have discussed the basics of training."),
        seg(5.0, 9.0, "So what happens when the model overfits"),
        seg(9.0, 15.0, "In that situation the model performs poorly on new data it has not seen."),
    ],
)

# Test 4: duplicate questions -> one normalized question
run(
    "Test 4: duplicates deduplicated",
    [
        seg(0.0, 4.0, "Let's first ask, what is machine learning?"),
        seg(30.0, 34.0, "So what exactly is machine learning anyway?"),
        seg(60.0, 65.0, "Machine learning is a method for learning patterns from data."),
    ],
)

# Test 5: filler questions excluded
run(
    "Test 5: filler excluded",
    [
        seg(0.0, 2.0, "Okay?"),
        seg(2.0, 4.0, "Are you ready?"),
        seg(4.0, 10.0, "Let's get started with today's lesson on neural networks."),
    ],
    expect_none=True,
)

# Test 6: multiple distinct questions, chronological order, real timestamps
result6 = run(
    "Test 6: multiple questions, real video-like transcript",
    [
        seg(0.0, 5.0, "Welcome back to the channel."),
        seg(5.0, 10.0, "Today, what is machine learning, and why does it matter?"),
        seg(10.0, 40.0, "Machine learning is a way for computers to learn from data without being explicitly programmed."),
        seg(40.0, 45.0, "It matters because it powers recommendation systems, translation, and more."),
        seg(340.0, 344.0, "Why is training data important?"),
        seg(344.0, 380.0, "Training data teaches the model what patterns to look for."),
        seg(760.0, 765.0, "Now, can AI replace programmers entirely?"),
        seg(765.0, 800.0, "Not entirely - AI assists programmers rather than replacing them."),
        seg(1080.0, 1086.0, "So what is overfitting exactly?"),
        seg(1086.0, 1120.0, "Overfitting is when a model memorizes training data instead of generalizing."),
        seg(1120.0, 1124.0, "Does that make sense?"),
        seg(1124.0, 1128.0, "Great, let's move on."),
    ],
)
if result6["success"]:
    starts = [q["start"] for q in result6["questions"]]
    print("chronological:", starts == sorted(starts))

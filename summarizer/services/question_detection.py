import logging
import re
import threading

from . import embeddings
from .summarization import (
    _clear_gpu_memory,
    _generate_batch,
    _is_out_of_memory,
    _load_model,
    generation_lock,
)
from .video_chat import format_timestamp, youtube_timestamp_url

logger = logging.getLogger(__name__)

# Batched per Mistral generate() call, reusing the already-loaded model.
# Decode steps (= VALIDATION_MAX_NEW_TOKENS) are paid once per batch, not per
# sequence in it, so a wider batch does the same total work in fewer sequential
# steps. Validation prompts are short (a 2-3 segment window) with a tiny
# 40-token output, so the KV cache per sequence is small - a much wider batch
# fits safely than for the longer chunk-summarization prompts.
VALIDATION_BATCH_SIZE = 20
VALIDATION_MAX_NEW_TOKENS = 40

# Hard cap on how many lightweight candidates are ever sent to Mistral, so a
# long interview-style video can't balloon the number of generate() calls.
# "?" candidates are prioritised over interrogative-start-only candidates.
MAX_CANDIDATES = 60
# Final cap on questions returned/stored per video, after dedup and ranking.
MAX_QUESTIONS = 12
# Cosine similarity (embeddings are L2-normalised, so dot product) above which
# two validated questions are treated as the same conceptual question.
DEDUP_SIMILARITY_THRESHOLD = 0.80

_GENERIC_ERROR = "We couldn't detect questions for this video."
_NONE_MARKER = "NONE"

_INTERROGATIVE_WORDS = (
    "what|why|how|when|where|who|whom|whose|which|"
    "can|could|should|would|will|"
    "do|does|did|is|are|was|were|has|have|had"
)
# A literal "?" is a strong signal on its own. An interrogative word only
# counts as a candidate at a clause start (optionally after a short filler
# like "so"/"now"/"but"), to avoid flagging every sentence that merely
# contains "is" or "do" somewhere in the middle.
_CLAUSE_START_RE = re.compile(
    rf"(?:^|[.!?]\s+|,\s*)(?:(?:so|now|but|and|then)\s+)?({_INTERROGATIVE_WORDS})\b",
    re.IGNORECASE,
)
_QUESTION_MARK_RE = re.compile(r"\?")

# Sentences that are ONLY conversational filler are dropped before ever
# reaching Mistral. Borderline cases are left for Mistral to judge in context.
_FILLER_RE = re.compile(
    r"^\s*(okay|alright|right|so|yeah|now)?[,\s]*"
    r"(are you (ready|there|following|with me|guys ready)|"
    r"can you (hear|see) me|"
    r"does (that|this) make sense|"
    r"you know|"
    r"right|"
    r"okay|"
    r"alright)"
    r"\s*\??\s*$",
    re.IGNORECASE,
)

_CONCEPTUAL_STARTERS = {"what", "why", "how", "can", "could", "should", "would"}

SYSTEM_PROMPT_QUESTION = (
    "You extract a question that is LITERALLY asked in a short excerpt of a video "
    "transcript. The excerpt often also contains the answer or surrounding context - "
    "extract ONLY the question part, lightly cleaned up (fix grammar/filler words only), "
    "ignoring the rest.\n\n"
    "Rules:\n"
    "- Only extract a question that is actually, literally stated in the excerpt. Do not "
    "compose, infer, or guess a question from the general topic - if you are not quoting "
    "something close to the excerpt's own words, you are inventing it, which is forbidden.\n"
    "- Do not add information, assumptions, or numbers that are not in the excerpt.\n"
    "- Skip small talk like 'okay?', 'are you ready?' or 'does that make sense?'.\n"
    "- If you are not confident the excerpt contains a real, literal question, respond "
    "NONE. When in doubt, respond NONE rather than guessing.\n"
    "- If there is no such question, respond with exactly: NONE\n"
    "- Respond with ONLY the question itself (ending in '?') or the word NONE. NEVER add "
    "a parenthetical note, assumption, explanation, or any other commentary about your "
    "answer - output nothing but the question or NONE.\n\n"
    "Example 1\n"
    "Excerpt: What is machine learning? Machine learning is a method for learning patterns "
    "from data.\n"
    "Answer: What is machine learning?\n\n"
    "Example 2\n"
    "Excerpt: Machine learning is a method for learning patterns from data.\n"
    "Answer: NONE\n\n"
    "Example 3\n"
    "Excerpt: Now we have discussed the basics. So what happens when the model overfits In "
    "that situation the model performs poorly on new data.\n"
    "Answer: What happens when the model overfits?\n\n"
    "Example 4\n"
    "Excerpt: Okay? Are you ready? Let's get started with today's lesson on neural "
    "networks.\n"
    "Answer: NONE"
)


def _is_pure_filler(sentence):
    return bool(_FILLER_RE.match(sentence.strip()))


def _sentence_is_candidate(sentence):
    sentence = sentence.strip()
    if not sentence or _is_pure_filler(sentence):
        return False
    if _QUESTION_MARK_RE.search(sentence):
        return True
    return bool(_CLAUSE_START_RE.search(sentence))


def _segment_has_candidate(text):
    """Split loosely on sentence-ending punctuation and check each piece -
    catches a question mid-segment even when most of the segment is not one.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _sentence_is_candidate(sentence):
            return True
    return False


# Whisper segments that are list-adjacent can still be far apart in time (it
# doesn't insert filler segments across a silence/scene-cut gap). Only merge
# consecutive candidate segments when the gap between them is small enough to
# indicate genuinely continuous speech - otherwise they're unrelated moments
# that just happen to sit next to each other in the list.
MAX_MERGE_GAP_SECONDS = 2.0


def _build_candidate_windows(segments):
    """Scan transcript segments for question candidates using lightweight
    keyword/punctuation signals (no Mistral call yet).

    Consecutive candidate segments separated by a small enough time gap are
    merged into one window (a question can run across a Whisper segment
    boundary); the anchor timestamp is the start of the first segment in that
    run, matching where the question begins. Each window's text includes one
    segment of context on either side so Mistral can judge a question that
    continues from the previous segment.

    Returns a list of {"text", "start", "end", "segment_index"}, in order.
    """
    flags = [_segment_has_candidate(segment["text"]) for segment in segments]

    windows = []
    i = 0
    while i < len(segments):
        if not flags[i]:
            i += 1
            continue

        run_start = i
        i += 1
        while (
            i < len(segments)
            and flags[i]
            and segments[i]["start"] - segments[i - 1]["end"] <= MAX_MERGE_GAP_SECONDS
        ):
            i += 1
        run_end = i - 1  # inclusive

        window_start_idx = max(0, run_start - 1)
        window_end_idx = min(len(segments) - 1, run_end + 1)
        text = " ".join(
            segments[j]["text"].strip() for j in range(window_start_idx, window_end_idx + 1)
        )

        windows.append(
            {
                "text": text,
                "start": segments[run_start]["start"],
                "end": segments[run_end]["end"],
                "segment_index": run_start,
            }
        )

    return windows


def _prioritize_candidates(windows, limit):
    """Keep the most promising `limit` candidates: an explicit '?' is a much
    stronger signal than an interrogative-word start alone.
    """
    def has_question_mark(window):
        return bool(_QUESTION_MARK_RE.search(window["text"]))

    ordered = sorted(windows, key=has_question_mark, reverse=True)
    dropped = len(windows) - limit
    if dropped > 0:
        logger.info("[QUESTIONS] %d candidates exceed the cap; dropping %d", len(windows), dropped)
    return ordered[:limit]


# The model occasionally echoes a label from the few-shot examples (e.g.
# "Answer: ..." or "Question: ...") instead of just the question itself.
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer|question|response)\s*:\s*", re.IGNORECASE)


def _normalize_question(text):
    text = _ANSWER_PREFIX_RE.sub("", text.strip())
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if not text.endswith("?"):
        text = text.rstrip(".!") + "?"
    return text[0].upper() + text[1:]


def _looks_like_a_question(text):
    return text.endswith("?") and 3 <= len(text.split()) <= 40


# The model hedging with its own uncertainty ("this assumes...", "not an
# exact question...") is a strong signal it's inventing rather than
# extracting - reject those outright regardless of what else they contain.
# Parenthetical asides in particular are rejected unconditionally: spoken
# questions in a transcript never naturally contain them, so any "(...)" in
# the model's answer is itself evidence of added commentary, whatever words
# it uses - a more general backstop than enumerating every hedge phrasing.
_HEDGE_RE = re.compile(
    r"[()]|not an exact question|this assumes|is not (?:just )?stat(?:ed|ing)",
    re.IGNORECASE,
)

# Minimum content-word overlap between a candidate question and the excerpt it
# was extracted from. Measured on real hallucinated output: fabricated
# questions scored 0.00-0.33 overlap with their source excerpt, while genuine
# extractions (including paraphrased ones) scored close to 1.00 - a wide,
# reliable gap. This is a mechanical backstop for "never hallucinate", not
# reliant on the model following instructions.
GROUNDING_OVERLAP_THRESHOLD = 0.5

_STOPWORDS = frozenset(
    "the a an is are was were do does did what why how when where who which can could "
    "should would will to of in on for and or that this it its we you i he she they "
    "them us be been being have has had not so but if then there here with from at as "
    "by your my his her our their".split()
)


def _content_words(text):
    return {word for word in re.findall(r"[a-z']+", text.lower()) if word not in _STOPWORDS and len(word) > 2}


def _is_grounded(question, excerpt):
    """Reject a question whose wording mostly doesn't appear anywhere in the
    excerpt it was supposedly extracted from - a mechanical check that the
    model actually quoted the transcript rather than composing a new question
    from the general topic.
    """
    question_words = _content_words(question)
    if not question_words:
        return False
    excerpt_words = _content_words(excerpt)
    overlap = len(question_words & excerpt_words) / len(question_words)
    return overlap >= GROUNDING_OVERLAP_THRESHOLD


def _validate_candidates(model, tokenizer, windows):
    """Send candidate windows to Mistral in batches, reusing the already
    loaded model. Returns validated {"question", "start", "end",
    "segment_index"} entries - only for windows Mistral confirmed contain a
    real, grounded question.
    """
    validated = []
    for offset in range(0, len(windows), VALIDATION_BATCH_SIZE):
        batch = windows[offset : offset + VALIDATION_BATCH_SIZE]
        logger.info(
            "[QUESTIONS] Validating candidates %d-%d of %d",
            offset + 1, offset + len(batch), len(windows),
        )
        messages_list = [
            [
                {"role": "system", "content": SYSTEM_PROMPT_QUESTION},
                {"role": "user", "content": f"Transcript excerpt:\n\n{window['text']}"},
            ]
            for window in batch
        ]
        outputs = _generate_batch(model, tokenizer, messages_list, VALIDATION_MAX_NEW_TOKENS)

        for window, raw_output in zip(batch, outputs):
            answer = _ANSWER_PREFIX_RE.sub("", raw_output.strip()).strip()
            if not answer or answer.upper().startswith(_NONE_MARKER) or _HEDGE_RE.search(answer):
                continue
            question = _normalize_question(answer.splitlines()[0])
            if not _looks_like_a_question(question) or _is_pure_filler(question):
                continue
            if not _is_grounded(question, window["text"]):
                logger.info(
                    "[QUESTIONS] Discarding ungrounded question (low overlap with source): %r",
                    question,
                )
                continue
            validated.append(
                {
                    "question": question,
                    "start": window["start"],
                    "end": window["end"],
                    "segment_index": window["segment_index"],
                }
            )

    return validated


def _deduplicate(questions):
    """Drop semantically-duplicate questions, keeping the earliest occurrence
    of each. `questions` must already be sorted chronologically. Uses the
    existing sentence-embedding service (already L2-normalised, so cosine
    similarity is a plain dot product) rather than another Mistral call.
    """
    if len(questions) <= 1:
        return list(questions)

    vectors = embeddings.embed_texts([q["question"] for q in questions])

    kept, kept_vectors = [], []
    for question, vector in zip(questions, vectors):
        is_duplicate = any(
            sum(a * b for a, b in zip(vector, other)) >= DEDUP_SIMILARITY_THRESHOLD
            for other in kept_vectors
        )
        if not is_duplicate:
            kept.append(question)
            kept_vectors.append(vector)

    return kept


def _relevance_score(question_text):
    """Internal-only heuristic used solely to pick which questions survive
    the MAX_QUESTIONS cap; never exposed to the API/UI."""
    words = question_text.rstrip("?").split()
    score = len(words)
    if words and words[0].lower() in _CONCEPTUAL_STARTERS:
        score += 5
    return score


def detect_questions(processed):
    """Detect meaningful, transcript-grounded questions from a Module 4
    processed-transcript structure (uses `processed["segments"]`).

    Returns {"success": True, "questions": [{"question", "start", "end",
    "segment_index"}, ...]} sorted chronologically, or
    {"success": False, "error": str} on failure. Never raises.
    """
    segments = processed.get("segments") or []
    if not segments:
        return {"success": True, "questions": []}

    try:
        windows = _build_candidate_windows(segments)
        if not windows:
            logger.info("[QUESTIONS] No lightweight candidates found")
            return {"success": True, "questions": []}

        if len(windows) > MAX_CANDIDATES:
            windows = _prioritize_candidates(windows, MAX_CANDIDATES)

        # Held around all Mistral usage: this may run in a background thread
        # (see detect_and_save_questions_async) alongside summarization, which
        # must not call generate() on the GPU at the same time as this does.
        with generation_lock:
            model, tokenizer = _load_model()
            validated = _validate_candidates(model, tokenizer, windows)
        validated.sort(key=lambda q: q["start"])

        deduped = _deduplicate(validated)

        if len(deduped) > MAX_QUESTIONS:
            deduped = sorted(deduped, key=lambda q: _relevance_score(q["question"]), reverse=True)
            deduped = deduped[:MAX_QUESTIONS]
        deduped.sort(key=lambda q: q["start"])

        logger.info(
            "[QUESTIONS] %d candidates -> %d validated -> %d final",
            len(windows), len(validated), len(deduped),
        )
        return {"success": True, "questions": deduped}
    except Exception as exc:
        if _is_out_of_memory(exc):
            logger.exception("[QUESTIONS] CUDA out of memory during question detection")
        else:
            logger.exception("[QUESTIONS] Failed to detect questions")
        return {"success": False, "error": _GENERIC_ERROR}
    finally:
        _clear_gpu_memory()


def detect_and_save_questions_async(video, processed):
    """Kick off question detection in a background thread and persist the
    result when it finishes, so the caller (the main summarize response)
    doesn't have to wait for it.

    This is why summarization is ~35-40s faster from the user's point of
    view: question detection still costs the same total GPU time (the shared
    generation_lock means it and summarization never actually run at once),
    but that time no longer sits on the critical path of the HTTP response.
    The frontend polls the (already-cached) summarize endpoint again a bit
    later to pick up the questions once this finishes.

    A video with no saved questions yet is therefore ambiguous - either
    detection hasn't finished, or the video genuinely has none - by design,
    since resolving that isn't needed: the UI simply shows nothing until (and
    unless) something arrives.

    Returns the started Thread (callers don't need it; tests use it to wait
    for completion instead of guessing at timing).
    """
    from .video_index import save_questions

    def _worker():
        try:
            result = detect_questions(processed)
        except Exception:
            logger.exception("[QUESTIONS] Background detection crashed for %s", video.video_id)
            return
        if result["success"]:
            save_questions(video, result["questions"])
        else:
            logger.error(
                "[QUESTIONS] Background detection failed for %s: %s", video.video_id, result["error"]
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def serialize_questions(video_id, questions):
    """Build the API-facing shape for a list of questions (either plain dicts
    from `detect_questions`, or VideoQuestion model instances from the cache).
    """
    result = []
    for index, question in enumerate(questions, start=1):
        if isinstance(question, dict):
            text, start, end = question["question"], question["start"], question["end"]
        else:
            text, start, end = question.question, question.start, question.end
        result.append(
            {
                "id": index,
                "question": text,
                "start": start,
                "end": end,
                "timestamp": format_timestamp(start),
                "youtube_url": youtube_timestamp_url(video_id, start),
            }
        )
    return result

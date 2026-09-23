import gc
import json
import logging
import re
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE_MAP = "auto"

# Generation parameters - kept in one place, not hardcoded throughout.
TEMPERATURE = 0.1
TOP_P = 0.95
DO_SAMPLE = True
# Decoding runs at only ~7 tokens/s for 4-bit Mistral on this 8 GB laptop GPU
# and is latency-bound (a batch of sequences costs about the same per step as
# one), so speed comes from generating fewer tokens and batching sequences.
CHUNK_MAX_NEW_TOKENS = 120
# The final answer is a detailed 15+ line summary plus short lists (~500
# tokens), so the cap must comfortably exceed that; it is only reached if the
# model rambles.
FINAL_MAX_NEW_TOKENS = 800
# Rendered summary length target: 15+ lines. At ~14 words per line in the
# result card, 15 lines is ~210 words, so ask for 4 paragraphs of 4-5 sentences.
MIN_SUMMARY_LINES = 15
# Chunks summarised per generate() call. Decode steps (= max_new_tokens) are
# paid once per batch, not per sequence in it, so fitting more chunks in one
# batch does the same total work in fewer sequential steps - on a real 13-chunk
# video, going from 2 batches of 8 to 1 batch of 13 roughly halved this stage.
# Each extra sequence adds ~130 KB of KV cache per token (8 x ~1K tokens ~ 1 GB
# measured), so this is bounded by VRAM, not usefulness. An out-of-memory batch
# is retried at half size.
CHUNK_BATCH_SIZE = 16
# At or below this many words the map step is skipped and the transcript is
# summarised directly - one generation instead of two.
DIRECT_SUMMARY_MAX_WORDS = 700

SYSTEM_PROMPT_CHUNK = (
    "You are an expert video summarization assistant.\n\n"
    "Rules:\n"
    "- Use only information present in the transcript excerpt below.\n"
    "- Do not invent facts or add information from outside the transcript.\n"
    "- Preserve important technical details, names, and numbers.\n"
    "- Remove repetition and filler words.\n"
    "- Focus on the main ideas.\n"
    "- Write {length_rule}.\n"
    "- Do not mention that you are an AI or describe what you are doing.\n"
    "- Respond with the summary only, no preamble or headings."
)

DEFAULT_CHUNK_LENGTH_RULE = "2 to 3 sentences, under about 70 words"


def chunk_length_spec(chunk_count):
    """(length rule, max new tokens) for each chunk summary.

    The final summary must have 15+ lines, which needs roughly 20 sentences of
    source material. With many chunks a couple of sentences each is plenty;
    with few chunks each one must carry more detail. Larger caps only cost
    time when the model actually uses them, and batches stay small for short
    videos.
    """
    if chunk_count >= 8:
        return DEFAULT_CHUNK_LENGTH_RULE, CHUNK_MAX_NEW_TOKENS
    if chunk_count >= 4:
        return "3 to 4 sentences, under about 100 words", 160
    return "5 to 6 sentences, under about 150 words", 240

_FINAL_PROMPT_TEMPLATE = (
    "You are an expert video summarization assistant.\n\n"
    "You will be given {input_description}\n\n"
    "Rules:\n"
    "- Combine the important information into one coherent summary.\n"
    "- Remove repetition.\n"
    "- Preserve important facts, names, and numbers.\n"
    "- Do not introduce any information that is not present in the input.\n"
    "- Do not guess or hallucinate.\n"
    "- Do not mention that you are an AI or describe the summarization process.\n"
    "- The summary must be DETAILED. Give it as the list \"summary_lines\" with AT LEAST "
    "15 entries, one sentence per entry (12 to 25 words), in the order things happen in the "
    "video. Every entry must state a different fact or point from the input; if the input is "
    "short, break it into finer details, but never invent, pad or repeat.\n"
    "- Keep the other fields brief: exactly 4 main topics (2-4 words each), exactly 4 key "
    "points and exactly 2 conclusions, each under 15 words.\n"
    "- Inside text values use single quotes only (never double quotes) and no backslashes.\n\n"
    "Format your response as a single JSON object with exactly these keys:\n"
    "{{\n"
    '  "summary_lines": ["first sentence", "second sentence", "third sentence", '
    '"... keep going until there are at least 15 entries"],\n'
    '  "main_topics": ["short topic phrase", "..."],\n'
    '  "key_points": ["key point", "..."],\n'
    '  "conclusions": ["conclusion", "..."]\n'
    "}}\n"
    "Respond with ONLY the JSON object and nothing else - no preamble, no code fences."
)

SYSTEM_PROMPT_FINAL = _FINAL_PROMPT_TEMPLATE.format(
    input_description="a series of section summaries from the same video, in order."
)
SYSTEM_PROMPT_DIRECT = _FINAL_PROMPT_TEMPLATE.format(
    input_description="the transcript of a video."
)

_LOAD_ERROR = "The summarization model could not be loaded. Please try again later."
_GENERATION_ERROR = "We couldn't generate a summary for this video. Please try again."
_EMPTY_INPUT_ERROR = "There is no transcript content to summarize."
_OOM_ERROR = (
    "The summarization model ran out of GPU memory. Please try a shorter video "
    "or try again later."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_TAIL_RE = re.compile(r"\{.*", re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Mistral escapes markdown underscores as "\_", which is not a valid JSON escape.
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')

_model = None
_tokenizer = None
_model_lock = threading.Lock()

# Serialises actual model.generate() usage across callers that may run on
# different threads (summarization's own map/reduce phases, and question
# detection running in a background thread alongside it). Concurrent
# generate() calls on one GPU don't speed anything up (compute still
# serialises) but DO double the resident KV-cache memory at once, which can
# OOM an 8 GB card - this lock keeps them one at a time instead.
generation_lock = threading.Lock()


def _load_model():
    """Lazily load and cache the Mistral model/tokenizer for reuse across
    requests within this process - the model is loaded once, not per chunk
    and not per request. Raises on failure; callers decide how to present
    that to the user. No automatic fallback to full precision or CPU.
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    with _model_lock:
        if _model is None or _tokenizer is None:
            logger.info("[MISTRAL] Loading model...")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map=DEVICE_MAP,
            )
            model.eval()

            _tokenizer = tokenizer
            _model = model

            logger.info("[MISTRAL] Model loaded successfully")
            logger.info("[MISTRAL] Device: %s", next(model.parameters()).device)

    return _model, _tokenizer


def preload_model_async():
    """Kick off model loading in a background thread if it isn't already
    loaded (or being loaded). Non-blocking - lets Mistral's load finish
    while earlier, independent pipeline stages (audio download, whisper
    transcription) are still running, instead of paying the full ~15-30s
    disk load only when summarization is first requested. Safe to call
    multiple times; a later call to `_load_model()` simply waits on the
    same lock rather than loading twice.
    """
    if _model is not None:
        return

    def _worker():
        try:
            _load_model()
        except Exception:
            logger.exception(
                "[MISTRAL] Background preload failed; will retry on first real request"
            )

    threading.Thread(target=_worker, daemon=True).start()


def _clear_gpu_memory():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _build_messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _generate(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output_ids[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    del inputs, output_ids, generated_tokens
    return text


def _is_out_of_memory(exc):
    """CUDA OOM surfaces as torch.OutOfMemoryError or, depending on where it
    happens, as a generic RuntimeError/AcceleratorError with "out of memory".
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _generate_batch(model, tokenizer, messages_list, max_new_tokens):
    """Generate one completion per conversation in `messages_list` using a
    single left-padded generate() call. Returns a list of strings, one per
    input, in order.
    """
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
        )

    texts = [
        tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
        for row in output_ids
    ]
    del inputs, output_ids
    return texts


def _chunk_messages(text, length_rule=DEFAULT_CHUNK_LENGTH_RULE):
    return _build_messages(
        SYSTEM_PROMPT_CHUNK.format(length_rule=length_rule),
        f"Summarize the following transcript excerpt:\n\n{text.strip()}",
    )


def _summarize_chunk_texts(model, tokenizer, texts, batch_timings):
    """Summarise every text, several per generate() call. Returns summaries
    aligned with `texts`. Texts of similar length are batched together to
    limit padding. If a batch runs out of GPU memory the batch size is halved
    and the batch retried, down to one at a time; only then does it raise.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    results = [None] * len(texts)
    batch_size = CHUNK_BATCH_SIZE
    position = 0
    length_rule, max_new_tokens = chunk_length_spec(len(texts))

    while position < len(order):
        indexes = order[position : position + batch_size]
        logger.info(
            "[MISTRAL] Summarizing chunks %s (batch of %d, %d/%d done)",
            sorted(i + 1 for i in indexes), len(indexes), position, len(order),
        )
        started = time.perf_counter()
        try:
            outputs = _generate_batch(
                model,
                tokenizer,
                [_chunk_messages(texts[i], length_rule) for i in indexes],
                max_new_tokens,
            )
        except Exception as exc:
            if _is_out_of_memory(exc) and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                logger.warning("[MISTRAL] Out of memory; retrying with batch size %d", batch_size)
                _clear_gpu_memory()
                continue
            raise

        for index, output in zip(indexes, outputs):
            results[index] = output
        batch_timings.append(
            {
                "seconds": round(time.perf_counter() - started, 1),
                "chunks": len(indexes),
                "input_words": sum(len(texts[i].split()) for i in indexes),
                "output_tokens": sum(
                    len(tokenizer.encode(o or "", add_special_tokens=False)) for o in outputs
                ),
            }
        )
        position += len(indexes)

    return results


def summarize_text(text, max_new_tokens=CHUNK_MAX_NEW_TOKENS):
    """Summarize a single piece of text with Mistral.

    Low-level building block reused both for per-chunk summarization and for
    ad-hoc/manual testing (see the Module 5 test script). Raises on model or
    generation failure - callers that need a clean, never-raise contract
    should use `summarize_transcript` instead, which wraps this.
    """
    if not text or not text.strip():
        raise ValueError("summarize_text requires non-empty text")

    model, tokenizer = _load_model()
    return _generate(model, tokenizer, _chunk_messages(text), max_new_tokens)


_CLOSER_FOR = {"{": "}", "[": "]"}


def _close_truncated_json(text):
    """Repair the bracket structure of near-valid JSON:

    - a mismatched closer gets the missing one inserted first (the model once
      closed an array with "}" instead of "]"), and
    - whatever is still open at the end (an unterminated string, then any open
      arrays/objects) is closed, for output cut off by the token cap or one
      character short of the final "}".
    """
    out, stack, in_string, escaped = [], [], False, False
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            out.append(char)
        elif char in "{[":
            stack.append(char)
            out.append(char)
        elif char in "}]":
            while stack and _CLOSER_FOR[stack[-1]] != char:
                out.append(_CLOSER_FOR[stack.pop()])
            if stack:
                stack.pop()
                out.append(char)
        else:
            out.append(char)

    closed = "".join(out) + ('"' if in_string else "")
    closed = re.sub(r",\s*$", "", closed)
    return closed + "".join(_CLOSER_FOR[opener] for opener in reversed(stack))


def _escape_inner_quotes(text):
    """Escape double quotes that appear *inside* JSON strings.

    Mistral quotes code in its text ("... -o "${F}.mkv" $F ..."), which ends
    the string early. A quote genuinely closes a string only if the next
    non-space character is one of , ] } : (or the text ends); any other quote
    inside a string is treated as content and escaped.
    """
    result, in_string, escaped = [], False, False
    for index, char in enumerate(text):
        if not in_string:
            result.append(char)
            if char == '"':
                in_string = True
        elif escaped:
            result.append(char)
            escaped = False
        elif char == "\\":
            result.append(char)
            escaped = True
        elif char == '"':
            rest = text[index + 1 :].lstrip()
            if not rest or rest[0] in ",]}:":
                result.append(char)
                in_string = False
            else:
                result.append('\\"')
        else:
            result.append(char)
    return "".join(result)


def _loads_lenient(text):
    """json.loads that tolerates invalid escapes, unescaped quotes inside
    strings, raw newlines in strings, and a truncated tail.
    """
    # strict=False: summary lines may contain raw newlines inside a string.
    cleaned = _INVALID_ESCAPE_RE.sub("", text)
    attempts = (
        lambda: cleaned,
        lambda: _close_truncated_json(cleaned),
        lambda: _escape_inner_quotes(cleaned),
        lambda: _close_truncated_json(_escape_inner_quotes(cleaned)),
    )
    error = None
    for build in attempts:
        try:
            return json.loads(build(), strict=False)
        except json.JSONDecodeError as exc:
            error = error or exc
    raise error


def _try_parse_json(text):
    try:
        return _loads_lenient(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_structured_summary(raw_text):
    """Try to parse Mistral's final-synthesis output as the structured JSON
    shape we asked for. Returns None if it doesn't look like valid,
    well-formed structured output - callers fall back to a plain summary
    rather than fighting to force structure out of the model.
    """
    candidates = [raw_text.strip()]
    fenced = _FENCED_BLOCK_RE.search(raw_text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        data = _try_parse_json(candidate)
        if data is None:
            # JSON embedded in prose: a complete object first, then one whose
            # closing brace is missing.
            for pattern in (_JSON_BLOCK_RE, _JSON_TAIL_RE):
                match = pattern.search(candidate)
                if match:
                    data = _try_parse_json(match.group(0))
                    if data is not None:
                        break

        if not isinstance(data, dict):
            continue

        def _as_str_list(value):
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()]

        # Preferred shape: "summary_lines" (a list, one sentence per line).
        # A plain "summary" string is still accepted.
        lines = _as_str_list(data.get("summary_lines"))
        summary = "\n".join(lines) if lines else data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            continue

        return {
            "summary": summary.strip(),
            "main_topics": _as_str_list(data.get("main_topics")),
            "key_points": _as_str_list(data.get("key_points")),
            "conclusions": _as_str_list(data.get("conclusions")),
        }

    return None


# Repeated at the very end of the prompt, where a 7B model follows it best: it
# otherwise wrote 8 key points / 6 conclusions, wasting ~20 s of decoding.
_FINAL_USER_SUFFIX = (
    "\n\nWrite the JSON object now: at least 15 summary_lines, then exactly 4 main_topics, "
    "exactly 4 key_points and exactly 2 conclusions. Single quotes only inside text."
)


def summary_line_count(summary):
    return len([line for line in str(summary).splitlines() if line.strip()])


_ENTRY_ROW_RE = re.compile(r'^"(.*)"\s*,?$')
_KEY_ROW_RE = re.compile(r'^"\w+"\s*:')


def _salvage_summary_lines(raw):
    """Last resort when the output can't be parsed as JSON at all: pull the
    "summary_lines" entries out row by row (the model prints one per row), so
    the user sees clean sentences instead of raw JSON. Returns a list or None.
    """
    entries, started = [], False
    for row in raw.splitlines():
        stripped = row.strip()
        if not started:
            started = '"summary_lines"' in stripped
            continue
        if stripped.startswith("]") or _KEY_ROW_RE.match(stripped):
            break
        match = _ENTRY_ROW_RE.match(stripped)
        if match and match.group(1).strip():
            entries.append(match.group(1).replace('\\"', '"').strip())
    return entries or None


def _final_attempt(model, tokenizer, messages):
    raw_output = _generate(model, tokenizer, messages, FINAL_MAX_NEW_TOKENS)

    structured = _parse_structured_summary(raw_output)
    if structured is not None:
        return structured

    salvaged = _salvage_summary_lines(raw_output)
    if salvaged:
        logger.warning(
            "[MISTRAL] Final summary was not parseable JSON; salvaged %d summary lines",
            len(salvaged),
        )
        return {
            "summary": "\n".join(salvaged),
            "main_topics": [],
            "key_points": [],
            "conclusions": [],
        }

    logger.info(
        "[MISTRAL] Final summary was not valid structured JSON; falling back to plain summary"
    )
    return {"summary": raw_output, "raw_output": raw_output}


def _generate_final_summary(model, tokenizer, combined_summaries, direct=False):
    """Produce the structured final summary, which must have at least
    MIN_SUMMARY_LINES lines. A 7B model often under-produces, so if the first
    attempt is short it is retried once with an explicit reminder and the
    longer of the two is kept (a hard guarantee is impossible, so a result
    that is still short after the retry is accepted rather than looped on).
    """
    if direct:
        messages = _build_messages(
            SYSTEM_PROMPT_DIRECT,
            "Transcript:\n\n" + combined_summaries + _FINAL_USER_SUFFIX,
        )
    else:
        messages = _build_messages(
            SYSTEM_PROMPT_FINAL,
            "Section summaries:\n\n" + combined_summaries + _FINAL_USER_SUFFIX,
        )

    result = _final_attempt(model, tokenizer, messages)
    lines = summary_line_count(result["summary"])
    if "raw_output" in result or lines >= MIN_SUMMARY_LINES:
        return result

    logger.info(
        "[MISTRAL] Summary has only %d lines (need %d); retrying once", lines, MIN_SUMMARY_LINES
    )
    reminder = (
        f"\n\nIMPORTANT: your previous answer had only {lines} entries in summary_lines. "
        f"Write at least {MIN_SUMMARY_LINES} entries, each a different fact from the input."
    )
    retry_messages = messages[:-1] + [
        {"role": "user", "content": messages[-1]["content"] + reminder}
    ]
    retry = _final_attempt(model, tokenizer, retry_messages)
    if "raw_output" not in retry and summary_line_count(retry["summary"]) > lines:
        return retry
    return result


def summarize_transcript(processed_transcript):
    """Summarize a Module 4 processed-transcript structure using Mistral:
    one summarization pass per chunk, followed by a final synthesis pass
    over the combined chunk summaries.

    Returns on success:
    {
        "success": True,
        "chunk_summaries": [{"chunk_id":1, "start":0.0, "end":298.6, "summary": "..."}, ...],
        "summary": "...",
        "main_topics": [...], "key_points": [...], "conclusions": [...],
        # or, if structured JSON generation was unreliable:
        # "summary": "...", "raw_output": "...",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "device": "cuda:0",
    }
    or {"success": False, "error": str} on failure. Never raises; technical
    exceptions (including CUDA OOM) are logged, never returned to the caller.
    """
    if not isinstance(processed_transcript, dict):
        logger.error(
            "summarize_transcript received a malformed input: %r", type(processed_transcript)
        )
        return {"success": False, "error": _EMPTY_INPUT_ERROR}

    chunks = processed_transcript.get("chunks") or []
    usable_chunks = [
        chunk for chunk in chunks if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
    ]

    if not usable_chunks:
        logger.error("summarize_transcript received no usable chunks")
        return {"success": False, "error": _EMPTY_INPUT_ERROR}

    load_started = time.perf_counter()
    try:
        model, tokenizer = _load_model()
    except torch.cuda.OutOfMemoryError:
        logger.exception("[MISTRAL] CUDA out of memory while loading the model")
        return {"success": False, "error": _OOM_ERROR}
    except Exception:
        logger.exception("[MISTRAL] Failed to load the model")
        return {"success": False, "error": _LOAD_ERROR}
    timings = {"model_wait": round(time.perf_counter() - load_started, 1), "batches": []}

    total_words = sum(len(str(chunk["text"]).split()) for chunk in usable_chunks)
    direct = total_words <= DIRECT_SUMMARY_MAX_WORDS
    timings["mode"] = "direct" if direct else "map_reduce"
    chunk_summaries = []

    # Held for the rest of this function: question detection may be running
    # concurrently in a background thread (see question_detection.py) and
    # must not call generate() on the GPU at the same time as this does.
    with generation_lock:
        if direct:
            logger.info("[MISTRAL] Short transcript (%d words): summarizing directly", total_words)
            combined = "\n\n".join(str(chunk["text"]).strip() for chunk in usable_chunks)
        else:
            try:
                summaries = _summarize_chunk_texts(
                    model, tokenizer, [str(chunk["text"]) for chunk in usable_chunks], timings["batches"]
                )
            except Exception as exc:
                if _is_out_of_memory(exc):
                    logger.exception("[MISTRAL] CUDA out of memory while summarizing chunks")
                    _clear_gpu_memory()
                    return {"success": False, "error": _OOM_ERROR}
                logger.exception("[MISTRAL] Failed to generate chunk summaries")
                return {"success": False, "error": _GENERATION_ERROR}

            for index, (chunk, summary_text) in enumerate(zip(usable_chunks, summaries), start=1):
                if not summary_text:
                    logger.warning("[MISTRAL] Chunk %d produced an empty summary; skipping", index)
                    continue
                chunk_summaries.append(
                    {
                        "chunk_id": chunk.get("chunk_id", index),
                        "start": chunk.get("start"),
                        "end": chunk.get("end"),
                        "summary": summary_text,
                    }
                )

            if not chunk_summaries:
                logger.error("[MISTRAL] All chunk summaries were empty or invalid")
                return {"success": False, "error": _GENERATION_ERROR}

        if not direct:
            combined = "\n\n".join(
                f"Section {entry['chunk_id']}: {entry['summary']}" for entry in chunk_summaries
            )

        logger.info("[MISTRAL] Generating final summary")
        final_started = time.perf_counter()
        try:
            final = _generate_final_summary(model, tokenizer, combined, direct=direct)
            timings["final_seconds"] = round(time.perf_counter() - final_started, 1)
            timings["summary_words"] = len(final["summary"].split())
            timings["summary_lines"] = summary_line_count(final["summary"])
        except Exception as exc:
            if _is_out_of_memory(exc):
                logger.exception("[MISTRAL] CUDA out of memory during final synthesis")
                _clear_gpu_memory()
                return {"success": False, "error": _OOM_ERROR}
            logger.exception("[MISTRAL] Failed to generate the final summary")
            return {"success": False, "error": _GENERATION_ERROR}

        _clear_gpu_memory()

    logger.info("[MISTRAL] Summarization completed")

    result = {
        "success": True,
        "chunk_summaries": chunk_summaries,
        "model": MODEL_ID,
        "device": str(next(model.parameters()).device),
        "timings": timings,
    }
    result.update(final)
    return result

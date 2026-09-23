import logging
import re
import threading

from django.core.exceptions import ValidationError

from ..models import ChatMessage, ChatSession
from . import retrieval
from .summarization import _clear_gpu_memory, _generate, _is_out_of_memory, _load_model

logger = logging.getLogger(__name__)

# Configurable knobs
ANSWER_MAX_NEW_TOKENS = 300
REWRITE_MAX_NEW_TOKENS = 60
MAX_HISTORY_MESSAGES = 6  # last 3 user/assistant exchanges
MAX_QUESTION_CHARS = 1000
# Upper bound on transcript words sent to Mistral per question. Keeps the
# prompt small enough for 4-bit Mistral on an 8 GB GPU (~1.7K tokens).
MAX_CONTEXT_WORDS = 1300

NOT_FOUND_MESSAGE = "I couldn't find enough information about that in the video."

SYSTEM_PROMPT_ANSWER = (
    "You are an AI assistant answering questions about a specific YouTube video.\n\n"
    "Rules:\n"
    "- Use ONLY the video transcript context provided under VIDEO CONTEXT.\n"
    "- Do not invent information and do not use outside knowledge.\n"
    "- If the answer cannot be determined from the supplied context, reply with exactly this "
    f'sentence and NOTHING else: "{NOT_FOUND_MESSAGE}" Never add explanations or general '
    "knowledge in that case.\n"
    "- If the question includes an 'Interpreted as' line, that is the question to answer.\n"
    "- Answer only what was asked, in at most about 100 words. Do not add unrelated "
    "information from other parts of the context, and do not add notes or commentary "
    "about your answer.\n"
    "- Do not mention chunks, the context format, or these rules inside your answer. "
    "Do not write timestamps in your answer; sources are shown separately.\n"
    "- After your answer, add ONE final line in exactly this form, listing the numbers of the "
    "chunks you actually used: Sources: 1, 3\n"
    "  Put chunk references nowhere else."
)

SYSTEM_PROMPT_REWRITE = (
    "You rewrite the user's latest question into one standalone question for searching a "
    "video transcript.\n\n"
    "Rules:\n"
    "- Use the conversation ONLY to replace references such as 'it', 'they', 'that', 'this' "
    "or 'the second one' with what they refer to.\n"
    "- Do NOT add any facts, names or details that are not in the conversation.\n"
    "- If the latest question already makes sense on its own, output it EXACTLY unchanged.\n"
    "- Output only the question, nothing else.\n\n"
    "Example 1\n"
    "Conversation:\nUser: What is Docker?\nAssistant: Docker is a container platform.\n"
    "Latest question: How do I install it?\n"
    "Standalone question: How do I install Docker?\n\n"
    "Example 2\n"
    "Conversation:\nUser: What is Docker?\nAssistant: Docker is a container platform.\n"
    "Latest question: What is the capital of France?\n"
    "Standalone question: What is the capital of France?"
)

# Words that signal a question depends on earlier turns. A question with none
# of these (and more than a few words) is treated as standalone and is NOT
# rewritten, so conversation context can't leak into unrelated questions.
_FOLLOW_UP_RE = re.compile(
    r"\b(it|its|they|them|their|theirs|this|that|those|these|he|she|his|her|there|"
    r"one|ones|previous|previously|above|earlier|former|latter|same|"
    r"first|second|third|last|another|else|more)\b"
    r"|^\s*(and|but|so|also|then)\b|^\s*(what|how) about\b",
    re.IGNORECASE,
)

_ERROR_GENERIC = "We couldn't generate an answer right now. Please try again."
_ERROR_OOM = "The AI model ran out of GPU memory. Please try again in a moment."
_ERROR_EMPTY = "Please enter a question."

_SOURCES_LINE_RE = re.compile(
    r"(?im)[ \t*]*\bsources?\b[ \t*]*:[ \t*]*"
    r"((?:chunks?[ \t]*)?\d+(?:[ \t]*(?:,|and|&)[ \t]*(?:chunks?[ \t]*)?\d+)*)"
    r"[ \t*]*\.?[ \t]*$"
)
# "Sources:" header on its own line followed by a list, through end of text.
_SOURCES_BLOCK_RE = re.compile(r"(?ims)^[ \t*#>-]*sources?[ \t*]*:[ \t*]*\n.*\Z")
# Parenthetical citations: "(Sources: 1, 3)", "(Source: Chunk 2)", "[Chunk 1]".
# A "source"/"chunk" word is required so ordinary "(1)" or "(2020)" is untouched.
_INLINE_CITE_RE = re.compile(
    r"\s*[(\[]\s*(?:sources?\s*:?\s*(?:chunks?\s*)?|chunks?\s*)"
    r"(\d+(?:\s*(?:,|and|&)\s*(?:chunks?\s*)?\d+)*)\s*[)\]]",
    re.IGNORECASE,
)

# Serialise generation so concurrent chat requests don't fight over the GPU.
_generation_lock = threading.Lock()


def format_timestamp(seconds):
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def youtube_timestamp_url(video_id, seconds):
    return f"https://www.youtube.com/watch?v={video_id}&t={max(0, int(seconds))}s"


def _build_sources(video, chunks):
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "similarity": chunk["similarity"],
            "label": f"{format_timestamp(chunk['start'])} – {format_timestamp(chunk['end'])}",
            "youtube_url": youtube_timestamp_url(video.video_id, chunk["start"]),
        }
        for chunk in chunks
    ]


def _fit_to_word_budget(chunks, max_words=MAX_CONTEXT_WORDS):
    """Keep the most relevant chunks (already ordered best-first) that fit in
    `max_words`. The best chunk is always kept, truncated at a word boundary if
    it alone exceeds the budget. Returns new dicts; input is not modified.
    """
    fitted, used = [], 0
    for chunk in chunks:
        words = chunk["text"].split()
        if fitted and used + len(words) > max_words:
            continue
        if len(words) > max_words:
            chunk = {**chunk, "text": " ".join(words[:max_words])}
            words = words[:max_words]
        fitted.append(chunk)
        used += len(words)
    return fitted


def _build_context(chunks):
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Chunk {index}]\n"
            f"Timestamp: {format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])}\n"
            f"Text:\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _extract_citations(answer, chunk_count):
    """Split Mistral's answer into (clean_answer, cited_chunk_indexes).

    The model is asked to end with "Sources: 1, 3" (1-based indexes into the
    context it was given). Those markers, and any inline "(Source: Chunk N)"
    it adds anyway, are removed from the visible text. Indexes outside the
    context are ignored, so a citation can never point at a chunk that was
    not actually retrieved.
    """
    cited = []

    def _collect(numbers_text):
        for number in re.findall(r"\d+", numbers_text):
            index = int(number)
            if 1 <= index <= chunk_count and index not in cited:
                cited.append(index)

    block = _SOURCES_BLOCK_RE.search(answer)
    if block:
        text = block.group(0)
        numbers = re.findall(r"(?i)chunk[ \t]*(\d+)", text) or re.findall(
            r"(?m)^[ \t]*(\d+)[.)]", text
        )
        _collect(" ".join(numbers))
        answer = answer[: block.start()]

    for match in _SOURCES_LINE_RE.finditer(answer):
        _collect(match.group(1))
    answer = _SOURCES_LINE_RE.sub("", answer)

    for match in _INLINE_CITE_RE.finditer(answer):
        _collect(match.group(1))
    answer = _INLINE_CITE_RE.sub("", answer)

    return answer.strip(), cited


def _looks_like_follow_up(question):
    return len(question.split()) <= 4 or bool(_FOLLOW_UP_RE.search(question))


def _rewrite_query(model, tokenizer, history, question):
    """Turn an ambiguous follow-up into a standalone retrieval query. Only
    runs for questions that look like follow-ups, and falls back to the
    original question on any problem, so a bad rewrite can never make things
    worse than not rewriting.
    """
    if not history or not _looks_like_follow_up(question):
        return question

    transcript = "\n".join(
        f"{'User' if message.role == ChatMessage.Role.USER else 'Assistant'}: {message.message[:300]}"
        for message in history
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REWRITE},
        {
            "role": "user",
            "content": f"Conversation:\n{transcript}\nLatest question: {question}\nStandalone question:",
        },
    ]

    try:
        rewritten = _generate(model, tokenizer, messages, REWRITE_MAX_NEW_TOKENS)
    except Exception as exc:
        if _is_out_of_memory(exc):
            raise
        logger.exception("[VIDEO CHAT] Query rewriting failed; using original question")
        return question

    rewritten = rewritten.strip().splitlines()[0].strip().strip('"') if rewritten.strip() else ""
    if not rewritten or len(rewritten) > 300:
        return question
    return rewritten


def get_or_create_session(video, session_id=None):
    """Return the ChatSession for `session_id` if it belongs to `video`,
    otherwise start a new one.
    """
    if session_id:
        try:
            session = ChatSession.objects.filter(id=session_id, video=video).first()
        except (ValueError, TypeError, ValidationError):
            session = None
        if session is not None:
            return session
    return ChatSession.objects.create(video=video)


def chat_with_video(session, question, top_k=retrieval.TOP_K, threshold=retrieval.SIMILARITY_THRESHOLD):
    """Answer `question` about `session.video`, grounded in retrieved
    transcript chunks, and persist both messages.

    Returns {"success": True, "answer", "sources", "session_id"} or
    {"success": False, "error"}. Never raises; technical errors are logged.
    """
    question = (question or "").strip()
    if not question:
        return {"success": False, "error": _ERROR_EMPTY}
    question = question[:MAX_QUESTION_CHARS]

    video = session.video
    logger.info("[VIDEO CHAT] Question received")

    history = list(session.messages.order_by("-created_at", "-id")[:MAX_HISTORY_MESSAGES])[::-1]

    try:
        with _generation_lock:
            model, tokenizer = _load_model()
            search_query = _rewrite_query(model, tokenizer, history, question)
            if search_query != question:
                logger.info("[VIDEO CHAT] Follow-up rewritten for retrieval")

            retrieved = retrieval.retrieve_chunks(video, search_query, top_k=top_k)
            relevant = _fit_to_word_budget(
                [chunk for chunk in retrieved if chunk["similarity"] >= threshold]
            )

            if not relevant:
                logger.info("[VIDEO CHAT] No chunk above similarity threshold; declining to answer")
                answer, sources = NOT_FOUND_MESSAGE, []
            else:
                # Prior answers are deliberately NOT sent: the standalone
                # rewrite already carries the conversation context, and
                # sending earlier answers made the model regurgitate them
                # instead of answering from the retrieved transcript.
                question_block = question
                if search_query != question:
                    question_block += f"\n(Interpreted as: {search_query})"
                user_content = (
                    f"USER QUESTION\n{question_block}\n\nVIDEO CONTEXT\n\n{_build_context(relevant)}"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
                    {"role": "user", "content": user_content},
                ]
                logger.info("[VIDEO CHAT] Generating Mistral response")
                answer = _generate(model, tokenizer, messages, ANSWER_MAX_NEW_TOKENS).strip()
                if not answer:
                    raise ValueError("Mistral returned an empty answer")
                answer, cited = _extract_citations(answer, len(relevant))
                if not answer:
                    raise ValueError("Mistral returned an empty answer")
                if NOT_FOUND_MESSAGE in answer:
                    # The model declined; discard anything else it appended
                    # (it may have added outside knowledge) and cite nothing.
                    answer, sources = NOT_FOUND_MESSAGE, []
                else:
                    used = [relevant[i - 1] for i in cited] or relevant
                    sources = _build_sources(video, used)
                logger.info("[VIDEO CHAT] Answer generated")
    except Exception as exc:
        if _is_out_of_memory(exc):
            logger.exception("[VIDEO CHAT] CUDA out of memory")
            return {"success": False, "error": _ERROR_OOM}
        logger.exception("[VIDEO CHAT] Failed to generate an answer")
        return {"success": False, "error": _ERROR_GENERIC}
    finally:
        # Return cached-but-unused VRAM after every answer so repeated chat
        # turns don't creep toward the 8 GB limit.
        _clear_gpu_memory()

    ChatMessage.objects.create(session=session, role=ChatMessage.Role.USER, message=question)
    ChatMessage.objects.create(
        session=session, role=ChatMessage.Role.ASSISTANT, message=answer, sources=sources
    )
    logger.info("[VIDEO CHAT] Sources: %d", len(sources))

    return {
        "success": True,
        "answer": answer,
        "sources": sources,
        "session_id": str(session.id),
    }

"""Tests for Chat with Video: embeddings, indexing, retrieval, chat service, endpoint.

The embedding model and Mistral are mocked (loading them per test run would be
slow and GPU-bound), but database access and pgvector similarity queries are real.
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from django.test import TestCase

from .models import EMBEDDING_DIMENSIONS, ChatMessage, ChatSession, TranscriptChunk, Video
from .services import embeddings as embeddings_module
from .services.retrieval import retrieve_chunks
from .services.video_chat import (
    NOT_FOUND_MESSAGE,
    _extract_citations,
    _fit_to_word_budget,
    _looks_like_follow_up,
    chat_with_video,
    format_timestamp,
    get_or_create_session,
    youtube_timestamp_url,
)
from .services.video_index import (
    SUMMARY_VERSION,
    get_cached_summary,
    get_indexed_video,
    index_video,
    save_summary,
)


def _unit_vector(index):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


def _processed(texts):
    chunks = [
        {
            "chunk_id": i,
            "start": i * 60.0,
            "end": i * 60.0 + 55.0,
            "text": text,
            "word_count": len(text.split()),
        }
        for i, text in enumerate(texts, start=1)
    ]
    return {"language": "en", "duration": 300.0, "full_text": " ".join(texts), "chunks": chunks}


class EmbeddingServiceTests(TestCase):
    def setUp(self):
        self._original = embeddings_module._model
        embeddings_module._model = None
        self.addCleanup(lambda: setattr(embeddings_module, "_model", self._original))

    @patch("summarizer.services.embeddings.SentenceTransformer")
    def test_embeds_texts_and_returns_lists(self, mock_class):
        mock_class.return_value.encode.return_value = np.array(
            [[0.1] * 384, [0.2] * 384], dtype=np.float32
        )
        vectors = embeddings_module.embed_texts(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)
        self.assertIsInstance(vectors[0], list)
        self.assertTrue(mock_class.return_value.encode.call_args.kwargs["normalize_embeddings"])

    @patch("summarizer.services.embeddings.SentenceTransformer")
    def test_model_loaded_once_on_configured_device(self, mock_class):
        mock_class.return_value.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        embeddings_module.embed_text("one")
        embeddings_module.embed_text("two")
        mock_class.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2", device=embeddings_module.DEVICE
        )

    @patch("summarizer.services.embeddings.SentenceTransformer")
    def test_empty_input_does_not_load_model(self, mock_class):
        self.assertEqual(embeddings_module.embed_texts([]), [])
        mock_class.assert_not_called()

    def test_embedding_dimension_matches_model_field(self):
        self.assertEqual(EMBEDDING_DIMENSIONS, 384)


@patch("summarizer.services.embeddings.embed_texts")
class VideoIndexTests(TestCase):
    def test_index_creates_video_transcript_and_chunks(self, mock_embed):
        mock_embed.return_value = [_unit_vector(0), _unit_vector(1)]
        result = index_video(
            "vid00000001",
            "https://youtu.be/vid00000001",
            {"title": "T", "channel": "C"},
            _processed(["first chunk text", "second chunk text"]),
        )
        self.assertTrue(result["success"])
        self.assertFalse(result["reused"])
        video = Video.objects.get(video_id="vid00000001")
        self.assertEqual(video.status, Video.Status.READY)
        self.assertEqual(video.title, "T")
        self.assertEqual(video.transcript.language, "en")
        self.assertEqual(video.chunks.count(), 2)
        chunk = video.chunks.get(chunk_id=2)
        self.assertEqual((chunk.start, chunk.end), (120.0, 175.0))
        self.assertEqual(chunk.text, "second chunk text")
        self.assertEqual(len(chunk.embedding), EMBEDDING_DIMENSIONS)

    def test_reindexing_reuses_embeddings(self, mock_embed):
        mock_embed.return_value = [_unit_vector(0)]
        index_video("vid00000001", "https://youtu.be/x", {}, _processed(["only chunk"]))
        again = index_video("vid00000001", "https://youtu.be/x", {}, _processed(["only chunk"]))
        self.assertTrue(again["reused"])
        mock_embed.assert_called_once()
        self.assertEqual(TranscriptChunk.objects.filter(video__video_id="vid00000001").count(), 1)

    def test_embedding_failure_marks_video_failed_and_stores_no_chunks(self, mock_embed):
        mock_embed.side_effect = RuntimeError("model exploded")
        result = index_video("vid00000001", "https://youtu.be/x", {}, _processed(["text"]))
        self.assertFalse(result["success"])
        self.assertNotIn("exploded", result["error"])
        video = Video.objects.get(video_id="vid00000001")
        self.assertEqual(video.status, Video.Status.FAILED)
        self.assertEqual(video.chunks.count(), 0)
        self.assertIsNone(get_indexed_video("vid00000001"))

    def test_no_chunks_fails_cleanly(self, mock_embed):
        result = index_video("vid00000001", "https://youtu.be/x", {}, {"chunks": []})
        self.assertFalse(result["success"])
        mock_embed.assert_not_called()

    def test_get_indexed_video_only_returns_ready_videos(self, mock_embed):
        self.assertIsNone(get_indexed_video("missing0001"))
        mock_embed.return_value = [_unit_vector(0)]
        index_video("vid00000001", "https://youtu.be/x", {}, _processed(["text"]))
        self.assertIsNotNone(get_indexed_video("vid00000001"))


@patch("summarizer.services.embeddings.embed_texts")
class SummaryCacheTests(TestCase):
    PAYLOAD = {"summary": "S", "main_topics": ["a"], "key_points": ["b"], "conclusions": []}

    def _index(self, mock_embed, video_id="vid00000001"):
        mock_embed.return_value = [_unit_vector(0)]
        index_video(
            video_id, f"https://youtu.be/{video_id}",
            {"title": "T", "thumbnail": "https://example.com/t.jpg"}, _processed(["text"]),
        )
        return Video.objects.get(video_id=video_id)

    def test_indexed_video_without_a_saved_summary_is_not_a_cache_hit(self, mock_embed):
        self._index(mock_embed)
        self.assertIsNone(get_cached_summary("vid00000001"))

    def test_saved_summary_is_returned_and_survives_a_round_trip(self, mock_embed):
        video = self._index(mock_embed)
        save_summary(video, self.PAYLOAD)
        cached = get_cached_summary("vid00000001")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.summary, {**self.PAYLOAD, "version": SUMMARY_VERSION})
        self.assertEqual(cached.thumbnail, "https://example.com/t.jpg")

    def test_summary_from_an_older_style_is_regenerated_not_served(self, mock_embed):
        video = self._index(mock_embed)
        # A pre-versioning (short-style) summary, and one from an older version.
        Video.objects.filter(pk=video.pk).update(summary=self.PAYLOAD)
        self.assertIsNone(get_cached_summary("vid00000001"))
        Video.objects.filter(pk=video.pk).update(summary={**self.PAYLOAD, "version": SUMMARY_VERSION - 1})
        self.assertIsNone(get_cached_summary("vid00000001"))

    def test_unknown_video_is_not_cached(self, mock_embed):
        self.assertIsNone(get_cached_summary("nothingthere"))

    def test_video_that_is_not_ready_is_not_served_from_cache(self, mock_embed):
        video = self._index(mock_embed)
        save_summary(video, self.PAYLOAD)
        Video.objects.filter(pk=video.pk).update(status=Video.Status.FAILED)
        self.assertIsNone(get_cached_summary("vid00000001"))

    def test_reindexing_does_not_discard_the_saved_summary(self, mock_embed):
        video = self._index(mock_embed)
        save_summary(video, self.PAYLOAD)
        self._index(mock_embed)
        self.assertEqual(
            get_cached_summary("vid00000001").summary, {**self.PAYLOAD, "version": SUMMARY_VERSION}
        )


@patch("summarizer.services.embeddings.embed_text")
class RetrievalTests(TestCase):
    def setUp(self):
        with patch("summarizer.services.embeddings.embed_texts") as mock_embed:
            mock_embed.return_value = [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
            index_video(
                "videoA00001", "https://youtu.be/a", {}, _processed(["ev", "pasta", "football"])
            )
            mock_embed.return_value = [_unit_vector(0)]
            index_video("videoB00001", "https://youtu.be/b", {}, _processed(["video b engines"]))
        self.video_a = Video.objects.get(video_id="videoA00001")

    def test_most_similar_chunk_is_first_with_score(self, mock_embed_text):
        mock_embed_text.return_value = _unit_vector(1)
        results = retrieve_chunks(self.video_a, "pasta question", top_k=3)
        self.assertEqual(results[0]["chunk_id"], 2)
        self.assertEqual(results[0]["text"], "pasta")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=3)
        self.assertLess(results[1]["similarity"], results[0]["similarity"])
        for key in ("chunk_id", "text", "start", "end", "similarity"):
            self.assertIn(key, results[0])

    def test_top_k_limits_results(self, mock_embed_text):
        mock_embed_text.return_value = _unit_vector(0)
        self.assertEqual(len(retrieve_chunks(self.video_a, "q", top_k=2)), 2)

    def test_retrieval_never_returns_other_videos_chunks(self, mock_embed_text):
        # The question vector is identical to Video B's only chunk.
        mock_embed_text.return_value = _unit_vector(0)
        results = retrieve_chunks(self.video_a, "engines?", top_k=10)
        self.assertEqual(len(results), 3)
        self.assertNotIn("video b engines", [r["text"] for r in results])


class ChatHelpersTests(TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(5), "00:05")
        self.assertEqual(format_timestamp(1122.5), "18:42")
        self.assertEqual(format_timestamp(3725), "1:02:05")

    def test_youtube_timestamp_url_uses_given_video_id_and_seconds(self):
        self.assertEqual(
            youtube_timestamp_url("abcDEF12345", 1122.5),
            "https://www.youtube.com/watch?v=abcDEF12345&t=1122s",
        )

    def test_extract_citations_removes_markers_and_maps_indexes(self):
        self.assertEqual(
            _extract_citations("Answer here. Sources: 1, 3.", 3), ("Answer here.", [1, 3])
        )
        self.assertEqual(_extract_citations("Answer. (Source: Chunk 2)", 3), ("Answer.", [2]))

    def test_extract_citations_handles_inline_parenthetical_citations(self):
        text = 'Use "-ss 10" (Sources: 1, 3). Then run it (Sources: 1).'
        self.assertEqual(_extract_citations(text, 5), ('Use "-ss 10". Then run it.', [1, 3]))

    def test_extract_citations_handles_trailing_sources_block(self):
        text = "Use Homebrew (Sources: 1).\n\nSources:\n1. Chunk 1 (00:00 - 02:52)\n2. Chunk 2 (30:03 - 33:05)"
        self.assertEqual(_extract_citations(text, 3), ("Use Homebrew.", [1, 2]))

    def test_extract_citations_leaves_ordinary_parentheses_and_prose_alone(self):
        text = "Version (1) shipped in (2020). Sources: wiki pages are mentioned."
        self.assertEqual(_extract_citations(text, 3), (text, []))

    def test_default_top_k_is_small_enough_for_8gb_gpu(self):
        from .services import retrieval

        self.assertEqual(retrieval.TOP_K, 3)

    def test_extract_citations_ignores_out_of_range_and_plain_numbers(self):
        self.assertEqual(
            _extract_citations("It costs 3 dollars.\nSources: 9", 2), ("It costs 3 dollars.", [])
        )

    def test_session_reused_only_for_same_video(self):
        video_a = Video.objects.create(video_id="videoA00001", url="https://youtu.be/a")
        video_b = Video.objects.create(video_id="videoB00001", url="https://youtu.be/b")
        session = get_or_create_session(video_a)
        self.assertEqual(get_or_create_session(video_a, str(session.id)).id, session.id)
        self.assertNotEqual(get_or_create_session(video_b, str(session.id)).id, session.id)
        self.assertNotEqual(get_or_create_session(video_a, "not-a-uuid").id, session.id)


class ChatWithVideoTests(TestCase):
    CHUNKS = [
        {"chunk_id": 4, "text": "Batteries store energy.", "start": 1122.5, "end": 1190.2, "similarity": 0.82},
        {"chunk_id": 9, "text": "Charging takes hours.", "start": 300.0, "end": 360.0, "similarity": 0.55},
    ]

    def setUp(self):
        self.video = Video.objects.create(video_id="videoA00001", url="https://youtu.be/a")
        self.session = ChatSession.objects.create(video=self.video)

        patches = {
            "load": patch(
                "summarizer.services.video_chat._load_model", return_value=(MagicMock(), MagicMock())
            ),
            "generate": patch("summarizer.services.video_chat._generate"),
            "retrieve": patch("summarizer.services.video_chat.retrieval.retrieve_chunks"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)
        self.mocks["retrieve"].return_value = [dict(c) for c in self.CHUNKS]

    def test_grounded_answer_returns_sources_from_cited_chunks_only(self):
        self.mocks["generate"].return_value = "They use batteries. Sources: 1"
        result = chat_with_video(self.session, "How is energy stored?")

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "They use batteries.")
        self.assertEqual(len(result["sources"]), 1)
        source = result["sources"][0]
        self.assertEqual(source["chunk_id"], 4)
        self.assertEqual(source["start"], 1122.5)
        self.assertEqual(source["label"], "18:42 – 19:50")
        self.assertEqual(
            source["youtube_url"], "https://www.youtube.com/watch?v=videoA00001&t=1122s"
        )

    def test_prompt_contains_question_context_and_real_timestamps(self):
        self.mocks["generate"].return_value = "Answer. Sources: 1"
        chat_with_video(self.session, "How is energy stored?")
        messages = self.mocks["generate"].call_args.args[2]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("ONLY", messages[0]["content"])
        final = messages[-1]["content"]
        self.assertIn("How is energy stored?", final)
        self.assertIn("[Chunk 1]", final)
        self.assertIn("Timestamp: 18:42 - 19:50", final)
        self.assertIn("Batteries store energy.", final)

    def test_messages_are_persisted_with_sources(self):
        self.mocks["generate"].return_value = "Answer. Sources: 2"
        chat_with_video(self.session, "Q?")
        messages = list(self.session.messages.all())
        self.assertEqual([m.role for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0].message, "Q?")
        self.assertEqual(messages[1].sources[0]["chunk_id"], 9)

    def test_uncited_answer_falls_back_to_all_retrieved_chunks(self):
        self.mocks["generate"].return_value = "An answer without any citation."
        result = chat_with_video(self.session, "Q?")
        self.assertEqual([s["chunk_id"] for s in result["sources"]], [4, 9])

    def test_weak_retrieval_declines_without_calling_mistral(self):
        self.mocks["retrieve"].return_value = [
            {"chunk_id": 1, "text": "unrelated", "start": 0.0, "end": 5.0, "similarity": 0.05}
        ]
        result = chat_with_video(self.session, "What is the capital of France?")
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.mocks["generate"].assert_not_called()

    def test_only_chunks_above_threshold_are_sent_to_mistral(self):
        self.mocks["generate"].return_value = "Answer. Sources: 1"
        chat_with_video(self.session, "Q?", threshold=0.7)
        final = self.mocks["generate"].call_args.args[2][-1]["content"]
        self.assertIn("Batteries store energy.", final)
        self.assertNotIn("Charging takes hours.", final)

    def test_model_declining_returns_no_sources(self):
        self.mocks["generate"].return_value = NOT_FOUND_MESSAGE
        result = chat_with_video(self.session, "Q?")
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertEqual(result["sources"], [])

    def test_follow_up_uses_history_and_rewritten_query_for_retrieval(self):
        ChatMessage.objects.create(session=self.session, role="user", message="What about Tesla?")
        ChatMessage.objects.create(
            session=self.session, role="assistant", message="Tesla makes EVs."
        )
        self.mocks["generate"].side_effect = [
            "What did the video say about Tesla's batteries?",
            "Answer. Sources: 1",
        ]

        result = chat_with_video(self.session, "What about its batteries?")

        self.assertTrue(result["success"])
        self.assertEqual(
            self.mocks["retrieve"].call_args.args[1],
            "What did the video say about Tesla's batteries?",
        )
        answer_messages = self.mocks["generate"].call_args_list[1].args[2]
        # Prior answers are NOT sent to the answering model (they made it
        # regurgitate earlier turns); the rewrite carries the context instead.
        self.assertEqual([m["role"] for m in answer_messages], ["system", "user"])
        final = answer_messages[-1]["content"]
        self.assertIn("What about its batteries?", final)
        self.assertIn("Interpreted as: What did the video say about Tesla's batteries?", final)
        self.assertNotIn("Tesla makes EVs.", final)

    def test_standalone_question_is_not_rewritten_even_with_history(self):
        # Regression: a rewriter once turned "What is the capital of France?"
        # into an FFmpeg question because of the earlier conversation.
        ChatMessage.objects.create(session=self.session, role="user", message="How do I trim video?")
        ChatMessage.objects.create(session=self.session, role="assistant", message="Use -ss.")
        self.mocks["generate"].return_value = "Answer. Sources: 1"
        chat_with_video(self.session, "What is the capital of France?")
        self.assertEqual(self.mocks["generate"].call_count, 1)
        self.assertEqual(
            self.mocks["retrieve"].call_args.args[1], "What is the capital of France?"
        )
        final = self.mocks["generate"].call_args.args[2][-1]["content"]
        self.assertNotIn("Interpreted as", final)

    def test_looks_like_follow_up(self):
        for question in ["What about its batteries?", "How do I use it?", "And with the fast one?",
                         "Why is that important?", "Really?", "what did they say about that"]:
            self.assertTrue(_looks_like_follow_up(question), question)
        for question in ["What is the capital of France?", "Explain how neural networks are trained.",
                         "How do I trim a video using ffmpeg?"]:
            self.assertFalse(_looks_like_follow_up(question), question)

    def test_decline_discards_any_outside_knowledge_the_model_appended(self):
        self.mocks["generate"].return_value = (
            f"{NOT_FOUND_MESSAGE} However, neural networks are trained with gradient descent. Sources: 1"
        )
        result = chat_with_video(self.session, "Explain neural networks?")
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)
        self.assertEqual(result["sources"], [])

    def test_default_similarity_threshold_is_calibrated(self):
        from .services import retrieval

        self.assertEqual(retrieval.SIMILARITY_THRESHOLD, 0.25)

    def test_first_question_is_not_rewritten(self):
        self.mocks["generate"].return_value = "Answer. Sources: 1"
        chat_with_video(self.session, "First question?")
        self.assertEqual(self.mocks["generate"].call_count, 1)
        self.assertEqual(self.mocks["retrieve"].call_args.args[1], "First question?")

    def test_failed_rewrite_falls_back_to_original_question(self):
        ChatMessage.objects.create(session=self.session, role="user", message="Earlier")
        self.mocks["generate"].side_effect = [RuntimeError("rewrite broke"), "Answer. Sources: 1"]
        result = chat_with_video(self.session, "Follow up?")
        self.assertTrue(result["success"])
        self.assertEqual(self.mocks["retrieve"].call_args.args[1], "Follow up?")

    def test_empty_question_is_rejected(self):
        result = chat_with_video(self.session, "   ")
        self.assertFalse(result["success"])
        self.mocks["retrieve"].assert_not_called()

    def test_generation_error_is_clean_and_saves_nothing(self):
        self.mocks["generate"].side_effect = RuntimeError("CUDA kernel panic")
        result = chat_with_video(self.session, "Q?")
        self.assertFalse(result["success"])
        self.assertNotIn("CUDA", result["error"])
        self.assertEqual(self.session.messages.count(), 0)

    def test_out_of_memory_error_is_reported_cleanly(self):
        self.mocks["generate"].side_effect = torch.OutOfMemoryError("CUDA OOM")
        result = chat_with_video(self.session, "Q?")
        self.assertFalse(result["success"])
        self.assertIn("GPU memory", result["error"])

    def test_generic_cuda_out_of_memory_runtime_error_is_also_recognised(self):
        # torch.AcceleratorError ("CUDA error: out of memory") is what a real
        # 8 GB overflow raised in testing; it is a RuntimeError, not OutOfMemoryError.
        self.mocks["generate"].side_effect = RuntimeError("CUDA error: out of memory")
        result = chat_with_video(self.session, "Q?")
        self.assertFalse(result["success"])
        self.assertIn("GPU memory", result["error"])
        self.assertEqual(self.session.messages.count(), 0)

    def test_oom_during_query_rewrite_is_not_swallowed(self):
        ChatMessage.objects.create(session=self.session, role="user", message="Earlier")
        self.mocks["generate"].side_effect = RuntimeError("CUDA error: out of memory")
        result = chat_with_video(self.session, "Follow up?")
        self.assertFalse(result["success"])
        self.assertIn("GPU memory", result["error"])

    def test_context_is_limited_to_word_budget(self):
        big = " ".join(["word"] * 1000)
        self.mocks["retrieve"].return_value = [
            {"chunk_id": 1, "text": big, "start": 0.0, "end": 60.0, "similarity": 0.9},
            {"chunk_id": 2, "text": big, "start": 60.0, "end": 120.0, "similarity": 0.8},
        ]
        self.mocks["generate"].return_value = "Answer. Sources: 1"
        chat_with_video(self.session, "Q?")
        final = self.mocks["generate"].call_args.args[2][-1]["content"]
        self.assertIn("[Chunk 1]", final)
        self.assertNotIn("[Chunk 2]", final)

    def test_oversized_single_chunk_is_truncated_not_dropped(self):
        huge = " ".join(["word"] * 5000)
        fitted = _fit_to_word_budget(
            [{"chunk_id": 1, "text": huge, "start": 0.0, "end": 1.0, "similarity": 0.9}]
        )
        self.assertEqual(len(fitted), 1)
        self.assertEqual(len(fitted[0]["text"].split()), 1300)


class ChatViewTests(TestCase):
    def post(self, payload, raw=False):
        body = payload if raw else json.dumps(payload)
        return self.client.post("/chat/", data=body, content_type="application/json")

    def index(self, video_id, text="some text"):
        with patch("summarizer.services.embeddings.embed_texts", return_value=[_unit_vector(0)]):
            index_video(video_id, f"https://youtu.be/{video_id}", {}, _processed([text]))

    def test_get_is_not_allowed(self):
        self.assertEqual(self.client.get("/chat/").status_code, 405)

    def test_invalid_json_is_rejected(self):
        response = self.post("{not json", raw=True)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])

    def test_empty_message_is_rejected(self):
        response = self.post({"video_id": "videoA00001", "message": "  "})
        self.assertEqual(response.status_code, 400)

    def test_unknown_or_unindexed_video_returns_404(self):
        Video.objects.create(
            video_id="videoA00001", url="https://youtu.be/a", status=Video.Status.FAILED
        )
        self.assertEqual(self.post({"video_id": "videoA00001", "message": "hi"}).status_code, 404)
        self.assertEqual(self.post({"video_id": "nope", "message": "hi"}).status_code, 404)

    @patch("summarizer.views.chat.chat_with_video")
    def test_successful_chat_returns_answer_and_sources(self, mock_chat):
        self.index("videoA00001")
        url = "https://www.youtube.com/watch?v=videoA00001&t=60s"
        mock_chat.return_value = {
            "success": True,
            "answer": "The answer.",
            "sources": [{"chunk_id": 1, "start": 60.0, "end": 115.0, "youtube_url": url}],
            "session_id": "abc",
        }
        response = self.post({"video_id": "videoA00001", "message": "hello?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "The answer.")
        self.assertEqual(data["sources"][0]["youtube_url"], url)
        session_arg, message_arg = mock_chat.call_args.args
        self.assertEqual(session_arg.video.video_id, "videoA00001")
        self.assertEqual(message_arg, "hello?")

    @patch("summarizer.views.chat.chat_with_video")
    def test_service_failure_returns_clean_error(self, mock_chat):
        self.index("videoA00001")
        mock_chat.return_value = {
            "success": False,
            "error": "We couldn't generate an answer right now. Please try again.",
        }
        response = self.post({"video_id": "videoA00001", "message": "hello?"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("Traceback", response.json()["error"])

    @patch("summarizer.views.chat.chat_with_video")
    def test_session_from_another_video_is_not_reused(self, mock_chat):
        self.index("videoA00001", "a text")
        self.index("videoB00001", "b text")
        session_a = ChatSession.objects.create(video=Video.objects.get(video_id="videoA00001"))
        mock_chat.return_value = {"success": True, "answer": "x", "sources": [], "session_id": "s"}
        self.post({"video_id": "videoB00001", "session_id": str(session_a.id), "message": "hi"})
        used_session = mock_chat.call_args.args[0]
        self.assertEqual(used_session.video.video_id, "videoB00001")
        self.assertNotEqual(used_session.id, session_a.id)

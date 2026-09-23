from pathlib import Path
import json
from unittest.mock import MagicMock, patch

import yt_dlp
from django.test import Client, TestCase

import torch

from .services import summarization as summarization_module
from .services import transcription as transcription_module
from .services.audio import extract_audio
from .services.summarization import _parse_structured_summary, summarize_transcript
from .services.transcript_processing import (
    DEFAULT_CHUNK_WORDS,
    chunk_transcript,
    clean_text,
    process_transcript,
    validate_and_clean_segments,
)
from .services.transcription import transcribe_audio
from .services.validators.youtube_url import validate_youtube_url
from .services.youtube import check_video_accessibility

VALID_ID = "dQw4w9WgXcQ"


class ValidateYoutubeUrlTests(TestCase):
    def test_accepts_standard_watch_url(self):
        result = validate_youtube_url(f"https://www.youtube.com/watch?v={VALID_ID}")
        self.assertTrue(result["valid"])
        self.assertEqual(result["video_id"], VALID_ID)

    def test_accepts_watch_url_without_www(self):
        result = validate_youtube_url(f"https://youtube.com/watch?v={VALID_ID}")
        self.assertTrue(result["valid"])

    def test_accepts_short_url(self):
        result = validate_youtube_url(f"https://youtu.be/{VALID_ID}")
        self.assertTrue(result["valid"])
        self.assertEqual(result["video_id"], VALID_ID)

    def test_accepts_watch_url_with_extra_params(self):
        result = validate_youtube_url(f"https://www.youtube.com/watch?v={VALID_ID}&t=120")
        self.assertTrue(result["valid"])
        self.assertEqual(result["video_id"], VALID_ID)

    def test_rejects_empty_url(self):
        self.assertFalse(validate_youtube_url("")["valid"])

    def test_rejects_random_website(self):
        self.assertFalse(validate_youtube_url("https://google.com")["valid"])

    def test_rejects_youtube_homepage(self):
        self.assertFalse(validate_youtube_url("https://youtube.com/")["valid"])

    def test_rejects_channel_url(self):
        self.assertFalse(validate_youtube_url("https://youtube.com/@channel")["valid"])

    def test_rejects_search_url(self):
        self.assertFalse(
            validate_youtube_url("https://youtube.com/results?search_query=test")["valid"]
        )

    def test_rejects_playlist_only_url(self):
        self.assertFalse(
            validate_youtube_url("https://youtube.com/playlist?list=XXXX")["valid"]
        )

    def test_rejects_malformed_url(self):
        self.assertFalse(validate_youtube_url("not-a-url")["valid"])

    def test_rejects_watch_url_without_video_id(self):
        self.assertFalse(validate_youtube_url("https://youtube.com/watch")["valid"])

    def test_rejects_none(self):
        self.assertFalse(validate_youtube_url(None)["valid"])


class CheckVideoAccessibilityTests(TestCase):
    @patch("summarizer.services.youtube.yt_dlp.YoutubeDL")
    def test_accessible_video_returns_metadata(self, mock_ydl_class):
        mock_ydl = mock_ydl_class.return_value.__enter__.return_value
        mock_ydl.extract_info.return_value = {
            "title": "Example Video",
            "channel": "Example Channel",
            "duration": 755,
            "thumbnail": "https://example.com/thumb.jpg",
        }

        result = check_video_accessibility(f"https://www.youtube.com/watch?v={VALID_ID}")

        self.assertTrue(result["accessible"])
        self.assertEqual(result["title"], "Example Video")
        self.assertEqual(result["channel"], "Example Channel")
        self.assertEqual(result["duration"], "12:35")

    @patch("summarizer.services.youtube.yt_dlp.YoutubeDL")
    def test_download_error_returns_clean_failure(self, mock_ydl_class):
        mock_ydl = mock_ydl_class.return_value.__enter__.return_value
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("Video unavailable")

        result = check_video_accessibility("https://www.youtube.com/watch?v=doesnotexist")

        self.assertFalse(result["accessible"])
        self.assertNotIn("Traceback", result["error"])
        self.assertNotIn("DownloadError", result["error"])

    @patch("summarizer.services.youtube.yt_dlp.YoutubeDL")
    def test_unexpected_exception_returns_clean_failure(self, mock_ydl_class):
        mock_ydl = mock_ydl_class.return_value.__enter__.return_value
        mock_ydl.extract_info.side_effect = RuntimeError("boom")

        result = check_video_accessibility(f"https://www.youtube.com/watch?v={VALID_ID}")

        self.assertFalse(result["accessible"])
        self.assertNotIn("boom", result["error"])


class ExtractAudioTests(TestCase):
    def setUp(self):
        from .services.audio import AUDIO_DIR

        self.audio_dir = AUDIO_DIR
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.video_id = "testvid123"
        self.addCleanup(self._cleanup_files)

    def _cleanup_files(self):
        for path in self.audio_dir.glob(f"{self.video_id}*"):
            path.unlink(missing_ok=True)

    def _mock_download(self, mock_ydl_class, ext="webm"):
        source_file = self.audio_dir / f"{self.video_id}.source.{ext}"
        source_file.write_bytes(b"fake source audio")
        mock_ydl = mock_ydl_class.return_value.__enter__.return_value
        mock_ydl.extract_info.return_value = {"id": self.video_id, "ext": ext}
        mock_ydl.prepare_filename.return_value = str(source_file)
        return source_file

    @patch("summarizer.services.audio.subprocess.run")
    @patch("summarizer.services.audio.yt_dlp.YoutubeDL")
    def test_successful_extraction_returns_wav_path(self, mock_ydl_class, mock_run):
        source_file = self._mock_download(mock_ydl_class)
        final_path = self.audio_dir / f"{self.video_id}.wav"

        def fake_ffmpeg(*args, **kwargs):
            final_path.write_bytes(b"RIFF....fake wav data")
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = fake_ffmpeg

        result = extract_audio(f"https://www.youtube.com/watch?v={self.video_id}", self.video_id)

        self.assertTrue(result["success"])
        self.assertEqual(result["path"], final_path)
        self.assertEqual(result["format"], "wav")
        self.assertEqual(result["sample_rate"], 16000)
        self.assertEqual(result["channels"], 1)
        self.assertTrue(final_path.exists())
        self.assertGreater(final_path.stat().st_size, 0)
        self.assertFalse(source_file.exists(), "temporary source file should be cleaned up")

    @patch("summarizer.services.audio.yt_dlp.YoutubeDL")
    def test_yt_dlp_download_failure_returns_clean_error(self, mock_ydl_class):
        mock_ydl = mock_ydl_class.return_value.__enter__.return_value
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("Video unavailable")

        result = extract_audio(f"https://www.youtube.com/watch?v={self.video_id}", self.video_id)

        self.assertFalse(result["success"])
        self.assertNotIn("Traceback", result["error"])
        self.assertNotIn("DownloadError", result["error"])

    @patch("summarizer.services.audio.subprocess.run")
    @patch("summarizer.services.audio.yt_dlp.YoutubeDL")
    def test_ffmpeg_failure_returns_clean_error_and_no_broken_file(
        self, mock_ydl_class, mock_run
    ):
        source_file = self._mock_download(mock_ydl_class)
        mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg: invalid data")

        result = extract_audio(f"https://www.youtube.com/watch?v={self.video_id}", self.video_id)

        self.assertFalse(result["success"])
        self.assertNotIn("ffmpeg", result["error"].lower())
        final_path = self.audio_dir / f"{self.video_id}.wav"
        self.assertFalse(final_path.exists())
        self.assertFalse(source_file.exists())

    @patch("summarizer.services.audio.subprocess.run")
    @patch("summarizer.services.audio.yt_dlp.YoutubeDL")
    def test_empty_output_file_treated_as_failure(self, mock_ydl_class, mock_run):
        self._mock_download(mock_ydl_class)
        final_path = self.audio_dir / f"{self.video_id}.wav"

        def fake_ffmpeg_empty(*args, **kwargs):
            final_path.touch()
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = fake_ffmpeg_empty

        result = extract_audio(f"https://www.youtube.com/watch?v={self.video_id}", self.video_id)

        self.assertFalse(result["success"])
        self.assertFalse(final_path.exists())


def _fake_segment(start, end, text):
    segment = MagicMock()
    segment.start = start
    segment.end = end
    segment.text = text
    return segment


def _fake_info(language="en", duration=10.0):
    info = MagicMock()
    info.language = language
    info.duration = duration
    return info


class TranscribeAudioTests(TestCase):
    def setUp(self):
        # The service caches its model in a module-level global; reset it
        # around each test so mocks don't leak between tests.
        self._original_model = transcription_module._model
        transcription_module._model = None
        self.addCleanup(self._restore_model)

    def _restore_model(self):
        transcription_module._model = self._original_model

    def test_missing_audio_file_returns_clean_failure(self):
        result = transcribe_audio("/nonexistent/path/audio.wav")

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Audio file could not be found.")

    @patch("summarizer.services.transcription.WhisperModel")
    def test_successful_transcription_returns_segments_and_language(self, mock_model_class):
        mock_model = mock_model_class.return_value
        mock_model.transcribe.return_value = (
            [
                _fake_segment(0.0, 5.2, "  Welcome to today's video.  "),
                _fake_segment(5.2, 12.8, "Today we discuss AI."),
            ],
            _fake_info(language="en", duration=12.8),
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = transcribe_audio("media/audio/fake.wav")

        self.assertTrue(result["success"])
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["duration"], 12.8)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(
            result["segments"][0],
            {"start": 0.0, "end": 5.2, "text": "Welcome to today's video."},
        )
        self.assertEqual(
            result["text"], "Welcome to today's video. Today we discuss AI."
        )
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["model"], "small")
        mock_model_class.assert_called_once_with("small", device="cuda", compute_type="float16")

    @patch("summarizer.services.transcription.WhisperModel")
    def test_every_segment_has_start_end_text(self, mock_model_class):
        mock_model = mock_model_class.return_value
        mock_model.transcribe.return_value = (
            [_fake_segment(0.0, 3.0, "one"), _fake_segment(3.0, 6.0, "two")],
            _fake_info(),
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = transcribe_audio("media/audio/fake.wav")

        for segment in result["segments"]:
            self.assertIn("start", segment)
            self.assertIn("end", segment)
            self.assertIn("text", segment)

    @patch("summarizer.services.transcription.WhisperModel")
    def test_language_is_not_hardcoded(self, mock_model_class):
        mock_model = mock_model_class.return_value
        mock_model.transcribe.return_value = (
            [_fake_segment(0.0, 2.0, "Hola")],
            _fake_info(language="es", duration=2.0),
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = transcribe_audio("media/audio/fake.wav")

        self.assertEqual(result["language"], "es")

    @patch("summarizer.services.transcription.WhisperModel")
    def test_model_is_loaded_once_and_reused(self, mock_model_class):
        mock_model = mock_model_class.return_value
        mock_model.transcribe.return_value = ([_fake_segment(0.0, 1.0, "hi")], _fake_info())

        with patch("pathlib.Path.exists", return_value=True):
            transcribe_audio("media/audio/fake.wav")
            transcribe_audio("media/audio/fake.wav")

        mock_model_class.assert_called_once()

    @patch("summarizer.services.transcription.WhisperModel")
    def test_cuda_init_failure_is_reported_cleanly_without_cpu_fallback(
        self, mock_model_class
    ):
        mock_model_class.side_effect = RuntimeError("CUDA driver error: no device")

        with patch("pathlib.Path.exists", return_value=True):
            result = transcribe_audio("media/audio/fake.wav")

        self.assertFalse(result["success"])
        self.assertNotIn("CUDA driver error", result["error"])
        # Only ever attempted with the configured (CUDA) device - no silent
        # CPU retry.
        mock_model_class.assert_called_once_with("small", device="cuda", compute_type="float16")

    @patch("summarizer.services.transcription.WhisperModel")
    def test_transcription_exception_returns_clean_failure(self, mock_model_class):
        mock_model = mock_model_class.return_value
        mock_model.transcribe.side_effect = RuntimeError("boom")

        with patch("pathlib.Path.exists", return_value=True):
            result = transcribe_audio("media/audio/fake.wav")

        self.assertFalse(result["success"])
        self.assertNotIn("boom", result["error"])

    def test_unload_model_releases_the_cached_model(self):
        transcription_module._model = MagicMock()
        transcription_module.unload_model()
        self.assertIsNone(transcription_module._model)

    def test_unload_model_is_safe_when_nothing_is_loaded(self):
        transcription_module._model = None
        transcription_module.unload_model()
        self.assertIsNone(transcription_module._model)

    @patch("summarizer.services.transcription.WhisperModel")
    def test_model_is_reloaded_lazily_after_unload(self, mock_model_class):
        mock_model_class.return_value.transcribe.return_value = ([], _fake_info())
        with patch("pathlib.Path.exists", return_value=True):
            transcribe_audio("media/audio/fake.wav")
            transcription_module.unload_model()
            transcribe_audio("media/audio/fake.wav")
        self.assertEqual(mock_model_class.call_count, 2)

    @patch("summarizer.services.transcription.threading.Thread")
    def test_preload_noop_when_model_already_loaded(self, mock_thread_class):
        transcription_module._model = MagicMock()
        transcription_module.preload_model_async()
        mock_thread_class.assert_not_called()

    @patch("summarizer.services.transcription.threading.Thread")
    def test_preload_starts_background_thread_when_not_loaded(self, mock_thread_class):
        transcription_module._model = None
        transcription_module.preload_model_async()
        mock_thread_class.assert_called_once()
        mock_thread_class.return_value.start.assert_called_once()


class TranscriptProcessingTests(TestCase):
    def test_clean_text_collapses_whitespace(self):
        self.assertEqual(clean_text("  Hello     everyone.  "), "Hello everyone.")

    def test_clean_text_collapses_newlines_and_tabs(self):
        self.assertEqual(clean_text("Hello\n\n\teveryone."), "Hello everyone.")

    def test_clean_text_empty_or_whitespace_returns_empty(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(clean_text(None), "")

    def test_clean_text_preserves_meaning_and_wording(self):
        text = "machine learning is a subset of artificial intelligence"
        self.assertEqual(clean_text(text), text)

    def test_whitespace_only_segment_is_ignored(self):
        segments = [{"start": 5, "end": 8, "text": "   "}]
        self.assertEqual(validate_and_clean_segments(segments), [])

    def test_empty_text_segment_is_ignored(self):
        segments = [{"start": 5, "end": 8, "text": ""}]
        self.assertEqual(validate_and_clean_segments(segments), [])

    def test_missing_text_segment_is_ignored(self):
        segments = [{"start": 5, "end": 8}]
        self.assertEqual(validate_and_clean_segments(segments), [])

    def test_missing_timestamp_segment_is_ignored(self):
        segments = [{"start": 5, "text": "hello"}]
        self.assertEqual(validate_and_clean_segments(segments), [])

    def test_end_before_start_segment_is_ignored(self):
        segments = [{"start": 8.0, "end": 5.0, "text": "hello"}]
        self.assertEqual(validate_and_clean_segments(segments), [])

    def test_non_numeric_timestamp_segment_is_ignored(self):
        segments = [{"start": "not-a-number", "end": 5.0, "text": "hello"}]
        self.assertEqual(validate_and_clean_segments(segments), [])

    def test_malformed_segment_does_not_crash_the_batch(self):
        segments = [
            None,
            {"start": 0.0, "end": 2.0, "text": "valid one"},
            "not a dict",
            {"start": 2.0, "end": 4.0, "text": "valid two"},
        ]
        cleaned = validate_and_clean_segments(segments)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0].text, "valid one")
        self.assertEqual(cleaned[1].text, "valid two")

    def test_timestamp_preservation_with_cleaning(self):
        segments = validate_and_clean_segments(
            [{"start": 10.5, "end": 15.8, "text": " Hello world "}]
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(
            segments[0].as_dict(), {"start": 10.5, "end": 15.8, "text": "Hello world"}
        )

    def test_full_text_combines_cleaned_segments(self):
        result = process_transcript(
            {
                "language": "en",
                "duration": 12.2,
                "segments": [
                    {"start": 0.0, "end": 5.4, "text": "  Welcome to today's video.  "},
                    {"start": 5.4, "end": 12.2, "text": "Today we discuss AI.  "},
                ],
            }
        )
        self.assertTrue(result["success"])
        self.assertEqual(
            result["full_text"],
            "Welcome to today's video. Today we discuss AI.",
        )
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["duration"], 12.2)

    def test_process_transcript_missing_segments_key_fails_cleanly(self):
        result = process_transcript({"language": "en", "duration": 1.0})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Transcript processing failed.")

    def test_process_transcript_non_dict_input_fails_cleanly(self):
        result = process_transcript(None)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Transcript processing failed.")

    def test_process_transcript_no_usable_segments_fails_cleanly(self):
        result = process_transcript(
            {"language": "en", "duration": 3.0, "segments": [{"start": 0, "end": 3, "text": "  "}]}
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "No usable transcript was found.")

    def test_chunking_keeps_segments_grouped_under_limit(self):
        segments = validate_and_clean_segments(
            [
                {"start": 0.0, "end": 1.0, "text": "one two three"},
                {"start": 1.0, "end": 2.0, "text": "four five six"},
            ]
        )
        chunks = chunk_transcript(segments, max_words=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["word_count"], 6)
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 2.0)

    def test_chunking_splits_when_limit_exceeded(self):
        segments = validate_and_clean_segments(
            [
                {"start": 0.0, "end": 1.0, "text": "one two three four five"},
                {"start": 1.0, "end": 2.0, "text": "six seven eight nine ten"},
            ]
        )
        chunks = chunk_transcript(segments, max_words=5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["word_count"], 5)
        self.assertEqual(chunks[1]["word_count"], 5)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[1]["chunk_id"], 2)

    def test_chunking_no_words_lost_or_duplicated(self):
        raw_segments = [
            {"start": float(i), "end": float(i + 1), "text": f"word{i}"} for i in range(50)
        ]
        segments = validate_and_clean_segments(raw_segments)
        chunks = chunk_transcript(segments, max_words=7)

        all_words = []
        for chunk in chunks:
            all_words.extend(chunk["text"].split())

        expected_words = [f"word{i}" for i in range(50)]
        self.assertEqual(all_words, expected_words)
        self.assertEqual(len(set(all_words)), len(all_words))

    def test_chunking_ids_are_sequential_and_have_required_fields(self):
        raw_segments = [
            {"start": float(i), "end": float(i + 1), "text": f"word{i}"} for i in range(20)
        ]
        segments = validate_and_clean_segments(raw_segments)
        chunks = chunk_transcript(segments, max_words=6)

        self.assertEqual([c["chunk_id"] for c in chunks], list(range(1, len(chunks) + 1)))
        for chunk in chunks:
            self.assertIn("start", chunk)
            self.assertIn("end", chunk)
            self.assertIn("text", chunk)
            self.assertIn("word_count", chunk)
            self.assertEqual(chunk["word_count"], len(chunk["text"].split()))

    def test_oversized_single_segment_is_split_not_dropped(self):
        huge_text = " ".join(f"word{i}" for i in range(1200))
        segments = validate_and_clean_segments(
            [{"start": 0.0, "end": 120.0, "text": huge_text}]
        )
        chunks = chunk_transcript(segments, max_words=DEFAULT_CHUNK_WORDS)

        total_words = sum(c["word_count"] for c in chunks)
        self.assertEqual(total_words, 1200)
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["word_count"], DEFAULT_CHUNK_WORDS)
            self.assertGreaterEqual(chunk["start"], 0.0)
            self.assertLessEqual(chunk["end"], 120.0)

    def test_process_transcript_end_to_end_with_chunks(self):
        segments = [
            {"start": float(i * 3), "end": float(i * 3 + 3), "text": f"  sentence number {i}  "}
            for i in range(300)
        ]
        result = process_transcript(
            {"language": "en", "duration": 900.0, "segments": segments}, max_chunk_words=100
        )

        self.assertTrue(result["success"])
        self.assertGreater(len(result["chunks"]), 1)
        total_words = sum(c["word_count"] for c in result["chunks"])
        self.assertEqual(total_words, len(result["full_text"].split()))


class SummarizationTests(TestCase):
    def setUp(self):
        self._original_model = summarization_module._model
        self._original_tokenizer = summarization_module._tokenizer
        summarization_module._model = None
        summarization_module._tokenizer = None
        self.batch_sizes = []
        self.batch_calls = []
        # Most tests use tiny one-line fixture summaries; disable the "retry if
        # under 15 lines" rule for them. The retry tests re-enable it explicitly.
        self.real_min_summary_lines = summarization_module.MIN_SUMMARY_LINES
        min_lines_patcher = patch("summarizer.services.summarization.MIN_SUMMARY_LINES", 1)
        min_lines_patcher.start()
        self.addCleanup(min_lines_patcher.stop)
        self.addCleanup(self._restore_model)

    def _restore_model(self):
        summarization_module._model = self._original_model
        summarization_module._tokenizer = self._original_tokenizer

    def _fake_batch(self, model, tokenizer, messages_list, max_new_tokens):
        """Stand-in for _generate_batch: 'summarises' each chunk by echoing its
        text, so results can be checked against the chunk they belong to
        regardless of how the chunks were ordered into batches."""
        self.batch_sizes.append(len(messages_list))
        self.batch_calls.append({"max_new_tokens": max_new_tokens, "system": messages_list[0][0]["content"]})
        return [
            "Summary of " + messages[1]["content"].split("\n\n", 1)[1].strip()
            for messages in messages_list
        ]

    def _mock_loaded_model(self, mock_model_class, mock_tokenizer_class):
        mock_model = mock_model_class.from_pretrained.return_value
        # parameters() is called more than once (load logging + final result),
        # so it must yield a fresh iterator each call, not a single exhausted one.
        mock_model.parameters.side_effect = lambda: iter([MagicMock(device="cuda:0")])
        mock_tokenizer = mock_tokenizer_class.from_pretrained.return_value
        return mock_model, mock_tokenizer

    # --- _parse_structured_summary ---

    def test_parse_structured_summary_valid_json(self):
        raw = (
            '{"summary": "A concise summary.", '
            '"main_topics": ["topic a", "topic b"], '
            '"key_points": ["point 1"], '
            '"conclusions": ["conclusion 1"]}'
        )
        result = _parse_structured_summary(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["summary"], "A concise summary.")
        self.assertEqual(result["main_topics"], ["topic a", "topic b"])
        self.assertEqual(result["key_points"], ["point 1"])
        self.assertEqual(result["conclusions"], ["conclusion 1"])

    def test_parse_structured_summary_handles_code_fence(self):
        raw = '```json\n{"summary": "Fenced summary.", "main_topics": [], "key_points": [], "conclusions": []}\n```'
        result = _parse_structured_summary(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["summary"], "Fenced summary.")

    def test_parse_structured_summary_handles_extra_prose_around_json(self):
        raw = 'Sure, here it is:\n{"summary": "Wrapped summary.", "main_topics": ["x"]}\nHope that helps!'
        result = _parse_structured_summary(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["summary"], "Wrapped summary.")

    def test_parse_structured_summary_recovers_when_final_brace_is_missing(self):
        # Real Mistral output: valid JSON, but it stopped before the closing "}".
        raw = (
            '{\n  "summary": "S.",\n  "main_topics": ["a", "b"],\n'
            '  "key_points": ["k1"],\n  "conclusions": ["c1"]'
        )
        result = _parse_structured_summary(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["main_topics"], ["a", "b"])
        self.assertEqual(result["conclusions"], ["c1"])

    def test_parse_structured_summary_recovers_from_a_cut_off_string(self):
        raw = '{"summary": "S", "main_topics": ["a"], "key_points": ["one", "tw'
        result = _parse_structured_summary(raw)
        self.assertEqual(result["summary"], "S")
        self.assertEqual(result["key_points"], ["one", "tw"])

    def test_parse_structured_summary_tolerates_markdown_escaped_underscores(self):
        raw = '{"summary": "Run ffmpeg -i input\\_file.mp4", "main_topics": []}'
        self.assertEqual(_parse_structured_summary(raw)["summary"], "Run ffmpeg -i input_file.mp4")

    def test_parse_structured_summary_recovers_unclosed_json_inside_prose(self):
        raw = 'Here you go: {"summary": "S", "main_topics": ["a"]'
        self.assertEqual(_parse_structured_summary(raw)["main_topics"], ["a"])

    def test_parse_structured_summary_invalid_json_returns_none(self):
        self.assertIsNone(_parse_structured_summary("This is just plain prose, not JSON."))

    def test_parse_structured_summary_missing_summary_key_returns_none(self):
        self.assertIsNone(_parse_structured_summary('{"main_topics": ["a"]}'))

    # --- summarize_transcript ---

    def test_summarize_transcript_non_dict_input_fails_cleanly(self):
        result = summarize_transcript(None)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "There is no transcript content to summarize.")

    def test_summarize_transcript_no_usable_chunks_fails_cleanly(self):
        result = summarize_transcript({"chunks": []})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "There is no transcript content to summarize.")

    def test_summarize_transcript_chunks_with_blank_text_fails_cleanly(self):
        result = summarize_transcript({"chunks": [{"chunk_id": 1, "text": "   "}]})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "There is no transcript content to summarize.")

    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_model_load_failure_returns_clean_error(self, mock_model_class, mock_tokenizer_class):
        mock_model_class.from_pretrained.side_effect = RuntimeError("no CUDA device")

        result = summarize_transcript(
            {"chunks": [{"chunk_id": 1, "start": 0.0, "end": 5.0, "text": "hello world"}]}
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["error"], "The summarization model could not be loaded. Please try again later."
        )
        self.assertNotIn("CUDA", result["error"])

    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_model_load_oom_returns_clean_oom_error(self, mock_model_class, mock_tokenizer_class):
        mock_model_class.from_pretrained.side_effect = torch.OutOfMemoryError("CUDA OOM")

        result = summarize_transcript(
            {"chunks": [{"chunk_id": 1, "start": 0.0, "end": 5.0, "text": "hello world"}]}
        )

        self.assertFalse(result["success"])
        self.assertIn("out of GPU memory", result["error"])

    # The map/reduce tests below force map mode (DIRECT_SUMMARY_MAX_WORDS=0);
    # otherwise these tiny fixture transcripts would be summarised directly.

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_successful_summarization_with_structured_output(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = self._fake_batch
        mock_generate.return_value = (
            '{"summary": "Overall summary.", "main_topics": ["a"], '
            '"key_points": ["b"], "conclusions": ["c"]}'
        )

        # Different lengths, so batching (which sorts by length) reorders them.
        processed = {
            "chunks": [
                {"chunk_id": 1, "start": 0.0, "end": 5.0, "text": "first chunk text"},
                {"chunk_id": 2, "start": 5.0, "end": 10.0, "text": "second chunk text here"},
            ]
        }
        result = summarize_transcript(processed)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["chunk_summaries"]), 2)
        # Summaries stay attached to the right chunk, in original order.
        self.assertEqual(result["chunk_summaries"][0]["summary"], "Summary of first chunk text")
        self.assertEqual(result["chunk_summaries"][0]["chunk_id"], 1)
        self.assertEqual(result["chunk_summaries"][0]["start"], 0.0)
        self.assertEqual(result["chunk_summaries"][1]["summary"], "Summary of second chunk text here")
        self.assertEqual(result["summary"], "Overall summary.")
        self.assertEqual(result["main_topics"], ["a"])
        self.assertEqual(result["model"], "mistralai/Mistral-7B-Instruct-v0.2")
        self.assertEqual(self.batch_sizes, [2])
        mock_generate.assert_called_once()
        self.assertEqual(result["timings"]["mode"], "map_reduce")
        self.assertEqual(len(result["timings"]["batches"]), 1)

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_falls_back_to_plain_summary_when_json_invalid(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = self._fake_batch
        mock_generate.return_value = "This is just a plain-text final summary, not JSON."

        processed = {"chunks": [{"chunk_id": 1, "start": 0.0, "end": 5.0, "text": "chunk text"}]}
        result = summarize_transcript(processed)

        self.assertTrue(result["success"])
        self.assertEqual(result["summary"], "This is just a plain-text final summary, not JSON.")
        self.assertEqual(result["raw_output"], "This is just a plain-text final summary, not JSON.")
        self.assertNotIn("main_topics", result)

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_oom_during_chunk_generation_returns_clean_error(
        self, mock_model_class, mock_tokenizer_class, mock_batch
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = torch.OutOfMemoryError("CUDA OOM during generate")

        processed = {"chunks": [{"chunk_id": 1, "start": 0.0, "end": 5.0, "text": "chunk text"}]}
        result = summarize_transcript(processed)

        self.assertFalse(result["success"])
        self.assertIn("out of GPU memory", result["error"])

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_all_empty_chunk_summaries_fails_cleanly(
        self, mock_model_class, mock_tokenizer_class, mock_batch
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.return_value = [""]

        processed = {"chunks": [{"chunk_id": 1, "start": 0.0, "end": 5.0, "text": "chunk text"}]}
        result = summarize_transcript(processed)

        self.assertFalse(result["success"])
        self.assertEqual(
            result["error"], "We couldn't generate a summary for this video. Please try again."
        )

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_model_is_loaded_once_for_multiple_chunks(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = self._fake_batch
        mock_generate.return_value = (
            '{"summary": "Final.", "main_topics": [], "key_points": [], "conclusions": []}'
        )

        processed = {
            "chunks": [
                {"chunk_id": i, "start": float(i), "end": float(i + 1), "text": f"chunk {i}"}
                for i in range(1, 4)
            ]
        }
        summarize_transcript(processed)

        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()

    # --- batching / speed ---

    @patch("summarizer.services.summarization.CHUNK_BATCH_SIZE", 2)
    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_chunks_are_summarised_in_batches_of_configured_size(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = self._fake_batch
        mock_generate.return_value = '{"summary": "S", "main_topics": [], "key_points": [], "conclusions": []}'

        processed = {
            "chunks": [
                {"chunk_id": i, "start": float(i), "end": float(i + 1), "text": "word " * i}
                for i in range(1, 6)
            ]
        }
        result = summarize_transcript(processed)

        self.assertEqual(self.batch_sizes, [2, 2, 1])
        self.assertEqual([c["chunk_id"] for c in result["chunk_summaries"]], [1, 2, 3, 4, 5])
        for chunk, summary in zip(processed["chunks"], result["chunk_summaries"]):
            self.assertEqual(summary["summary"], "Summary of " + chunk["text"].strip())

    @patch("summarizer.services.summarization.CHUNK_BATCH_SIZE", 4)
    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_out_of_memory_halves_batch_size_and_retries(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)

        def flaky(model, tokenizer, messages_list, max_new_tokens):
            if len(messages_list) > 2:
                raise RuntimeError("CUDA error: out of memory")
            return self._fake_batch(model, tokenizer, messages_list, max_new_tokens)

        mock_batch.side_effect = flaky
        mock_generate.return_value = '{"summary": "S", "main_topics": [], "key_points": [], "conclusions": []}'

        processed = {
            "chunks": [
                {"chunk_id": i, "start": float(i), "end": float(i + 1), "text": f"chunk {i}"}
                for i in range(1, 5)
            ]
        }
        result = summarize_transcript(processed)

        self.assertTrue(result["success"])
        self.assertEqual(self.batch_sizes, [2, 2])
        self.assertEqual(len(result["chunk_summaries"]), 4)

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_out_of_memory_at_batch_size_one_returns_clean_error(
        self, mock_model_class, mock_tokenizer_class, mock_batch
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = RuntimeError("CUDA error: out of memory")

        processed = {
            "chunks": [
                {"chunk_id": i, "start": 0.0, "end": 1.0, "text": f"chunk {i}"} for i in range(1, 4)
            ]
        }
        result = summarize_transcript(processed)

        self.assertFalse(result["success"])
        self.assertIn("out of GPU memory", result["error"])

    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_short_transcript_is_summarised_directly_in_one_generation(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_generate.return_value = (
            '{"summary": "Short video.", "main_topics": ["x"], "key_points": [], "conclusions": []}'
        )

        processed = {
            "chunks": [{"chunk_id": 1, "start": 0.0, "end": 60.0, "text": "A short transcript."}]
        }
        result = summarize_transcript(processed)

        self.assertTrue(result["success"])
        self.assertEqual(result["summary"], "Short video.")
        mock_batch.assert_not_called()
        mock_generate.assert_called_once()
        messages = mock_generate.call_args.args[2]
        self.assertIn("transcript of a video", messages[0]["content"])
        self.assertIn("A short transcript.", messages[1]["content"])
        self.assertEqual(result["chunk_summaries"], [])
        self.assertEqual(result["timings"]["mode"], "direct")

    # --- detailed (15+ line) summary ---

    def test_final_prompts_demand_a_list_of_at_least_15_summary_lines(self):
        for prompt in (summarization_module.SYSTEM_PROMPT_FINAL, summarization_module.SYSTEM_PROMPT_DIRECT):
            self.assertIn("summary_lines", prompt)
            self.assertIn("AT LEAST 15 entries", prompt)
            self.assertIn("never invent, pad or repeat", prompt)

    def test_the_real_minimum_is_fifteen_lines(self):
        # (MIN_SUMMARY_LINES is patched to 1 during tests; this is the real value.)
        self.assertEqual(self.real_min_summary_lines, 15)

    def test_summary_lines_list_is_joined_one_line_per_entry(self):
        raw = json.dumps({
            "summary_lines": ["One.", "Two.", "Three."],
            "main_topics": ["a"], "key_points": [], "conclusions": [],
        })
        result = _parse_structured_summary(raw)
        self.assertEqual(result["summary"], "One.\nTwo.\nThree.")
        self.assertEqual(summarization_module.summary_line_count(result["summary"]), 3)

    def test_summary_lines_is_preferred_but_plain_summary_still_accepted(self):
        both = json.dumps({"summary_lines": ["From list."], "summary": "From string."})
        self.assertEqual(_parse_structured_summary(both)["summary"], "From list.")
        only_string = json.dumps({"summary": "From string.", "main_topics": []})
        self.assertEqual(_parse_structured_summary(only_string)["summary"], "From string.")
        empty_list = json.dumps({"summary_lines": [], "summary": "Fallback string."})
        self.assertEqual(_parse_structured_summary(empty_list)["summary"], "Fallback string.")

    def test_parse_real_output_with_unescaped_quotes_in_code_and_a_cut_off_tail(self):
        # Reproduced from a real run on a 53-minute FFmpeg tutorial: the model
        # quoted a shell command with raw double quotes, used markdown-escaped
        # characters, then hit the token cap in the middle of a key point.
        raw = (
            '{\n  "summary_lines": [\n'
            '    "The video explains how to install FFMpeg through Homebrew.",\n'
            "    \"To convert a directory, run the command 'ffmpeg -i F -c:v libx264 "
            '-o "${F%.\\*}.mkv" $F\'.",\n'
            '    "Both methods give the same output but differ in speed."\n'
            '  ],\n  "main_topics": ["Installing FFMpeg", "Trimming videos"],\n'
            '  "key_points": ["Installing Homebrew", "Using \'-ss\' and \'-t\' flags", "Using FFm'
        )
        result = _parse_structured_summary(raw)

        self.assertIsNotNone(result, "must not fall back to showing raw JSON")
        lines = result["summary"].split("\n")
        self.assertEqual(len(lines), 3)
        self.assertIn('-o "${F%.*}.mkv" $F', lines[1])
        self.assertEqual(result["main_topics"], ["Installing FFMpeg", "Trimming videos"])
        self.assertEqual(result["key_points"][-1], "Using FFm")

    def test_parse_real_output_where_an_array_was_closed_with_a_curly_brace(self):
        # Reproduced from a real run: the model ended "conclusions" with "}"
        # instead of "]" ("...syntax\"\n}"), leaving the array unclosed.
        raw = (
            '{\n  "summary_lines": [\n    "Line one.",\n    "Line two."\n  ],\n'
            '  "main_topics": ["Topic A", "Topic B"],\n'
            '  "key_points": ["Point 1", "Point 2"],\n'
            '  "conclusions": ["Conclusion 1", "Conclusion 2 requiring correction"\n}'
        )
        result = _parse_structured_summary(raw)
        self.assertIsNotNone(result, "must not fall back to showing raw JSON")
        self.assertEqual(result["summary"], "Line one.\nLine two.")
        self.assertEqual(result["conclusions"], ["Conclusion 1", "Conclusion 2 requiring correction"])

    def test_close_truncated_json_repairs_mismatched_and_missing_closers(self):
        fix = summarization_module._close_truncated_json
        self.assertEqual(json.loads(fix('{"a": [1, 2}')), {"a": [1, 2]})
        self.assertEqual(json.loads(fix('{"a": {"b": [1, 2')), {"a": {"b": [1, 2]}})
        self.assertEqual(json.loads(fix('{"a": "cut o')), {"a": "cut o"})
        valid = '{"a": [1, {"b": 2}], "c": "x"}'
        self.assertEqual(fix(valid), valid)

    def test_unparseable_json_still_yields_clean_summary_lines_not_raw_json(self):
        # Mangled beyond repair (an unquoted key), but the entries are one per row.
        raw = (
            '{\n  "summary_lines": [\n    "First point.",\n'
            '    "Second point with a \\"quote\\" inside.",\n    "Third point."\n  ],\n'
            '  main_topics: ["oops", unquoted],\n  "key_points": ["k"\n}'
        )
        self.assertIsNone(_parse_structured_summary(raw))  # genuinely unparseable
        with patch("summarizer.services.summarization._generate", return_value=raw):
            result = summarization_module._final_attempt(MagicMock(), MagicMock(), [])
        self.assertNotIn("raw_output", result)
        self.assertEqual(
            result["summary"], 'First point.\nSecond point with a "quote" inside.\nThird point.'
        )
        self.assertNotIn("{", result["summary"])
        self.assertEqual(result["main_topics"], [])

    def test_salvage_returns_none_when_there_are_no_summary_lines(self):
        self.assertIsNone(summarization_module._salvage_summary_lines("just some prose"))
        self.assertIsNone(summarization_module._salvage_summary_lines('{"summary_lines": []}'))

    def test_final_user_prompt_repeats_the_count_constraints_at_the_end(self):
        with patch("summarizer.services.summarization._generate") as mock_generate:
            mock_generate.return_value = self._lines_json(3)
            summarization_module._generate_final_summary(MagicMock(), MagicMock(), "material", direct=True)
        user_prompt = mock_generate.call_args.args[2][-1]["content"]
        self.assertTrue(user_prompt.endswith("Single quotes only inside text."))
        self.assertIn("exactly 4 key_points", user_prompt)

    def test_escape_inner_quotes_only_touches_quotes_inside_strings(self):
        fixed = summarization_module._escape_inner_quotes('{"a": "say "hi" now", "b": ["x", "y"]}')
        self.assertEqual(json.loads(fixed), {"a": 'say "hi" now', "b": ["x", "y"]})
        untouched = '{"a": "plain", "b": ["x"]}'
        self.assertEqual(summarization_module._escape_inner_quotes(untouched), untouched)

    def test_summary_line_count_ignores_blank_lines(self):
        self.assertEqual(summarization_module.summary_line_count("a\n\n b \n   \nc"), 3)

    def _lines_json(self, count):
        return json.dumps({
            "summary_lines": [f"Sentence number {i}." for i in range(1, count + 1)],
            "main_topics": ["t"], "key_points": ["k"], "conclusions": ["c"],
        })

    def test_short_summary_is_retried_once_and_the_longer_result_kept(self):
        model, tokenizer = MagicMock(), MagicMock()
        with patch("summarizer.services.summarization.MIN_SUMMARY_LINES", 15), patch(
            "summarizer.services.summarization._generate"
        ) as mock_generate:
            mock_generate.side_effect = [self._lines_json(6), self._lines_json(16)]
            result = summarization_module._generate_final_summary(model, tokenizer, "material", direct=True)

        self.assertEqual(mock_generate.call_count, 2)
        self.assertEqual(summarization_module.summary_line_count(result["summary"]), 16)
        retry_prompt = mock_generate.call_args_list[1].args[2][-1]["content"]
        self.assertIn("only 6 entries", retry_prompt)
        self.assertIn("at least 15", retry_prompt)

    def test_retry_that_is_not_longer_keeps_the_original(self):
        with patch("summarizer.services.summarization.MIN_SUMMARY_LINES", 15), patch(
            "summarizer.services.summarization._generate"
        ) as mock_generate:
            mock_generate.side_effect = [self._lines_json(8), self._lines_json(5)]
            result = summarization_module._generate_final_summary(MagicMock(), MagicMock(), "m")
        self.assertEqual(summarization_module.summary_line_count(result["summary"]), 8)
        self.assertEqual(mock_generate.call_count, 2)

    def test_long_enough_summary_is_not_retried(self):
        with patch("summarizer.services.summarization.MIN_SUMMARY_LINES", 15), patch(
            "summarizer.services.summarization._generate"
        ) as mock_generate:
            mock_generate.return_value = self._lines_json(15)
            summarization_module._generate_final_summary(MagicMock(), MagicMock(), "m")
        mock_generate.assert_called_once()

    def test_unparseable_output_is_not_retried(self):
        with patch("summarizer.services.summarization.MIN_SUMMARY_LINES", 15), patch(
            "summarizer.services.summarization._generate"
        ) as mock_generate:
            mock_generate.return_value = "plain prose, not JSON"
            result = summarization_module._generate_final_summary(MagicMock(), MagicMock(), "m")
        mock_generate.assert_called_once()
        self.assertIn("raw_output", result)

    def test_final_token_cap_leaves_room_for_a_long_summary(self):
        self.assertGreaterEqual(summarization_module.FINAL_MAX_NEW_TOKENS, 600)

    def test_chunk_detail_scales_up_when_there_are_few_chunks(self):
        many_rule, many_tokens = summarization_module.chunk_length_spec(12)
        some_rule, some_tokens = summarization_module.chunk_length_spec(5)
        few_rule, few_tokens = summarization_module.chunk_length_spec(2)
        self.assertLess(many_tokens, some_tokens)
        self.assertLess(some_tokens, few_tokens)
        self.assertIn("2 to 3 sentences", many_rule)
        self.assertIn("5 to 6 sentences", few_rule)

    @patch("summarizer.services.summarization.DIRECT_SUMMARY_MAX_WORDS", 0)
    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization._generate_batch")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_few_chunks_are_asked_for_more_detail_than_many_chunks(
        self, mock_model_class, mock_tokenizer_class, mock_batch, mock_generate
    ):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_batch.side_effect = self._fake_batch
        mock_generate.return_value = '{"summary": "S", "main_topics": [], "key_points": [], "conclusions": []}'

        def run(count):
            self.batch_calls.clear()
            summarize_transcript(
                {"chunks": [{"chunk_id": i, "start": 0.0, "end": 1.0, "text": f"chunk {i}"} for i in range(1, count + 1)]}
            )
            return self.batch_calls[0]

        few, many = run(2), run(10)
        self.assertGreater(few["max_new_tokens"], many["max_new_tokens"])
        self.assertIn("5 to 6 sentences", few["system"])
        self.assertIn("2 to 3 sentences", many["system"])

    def test_parse_multi_paragraph_summary_with_escaped_and_raw_newlines(self):
        escaped = '{"summary": "Para one.\\n\\nPara two.", "main_topics": ["a"]}'
        self.assertEqual(_parse_structured_summary(escaped)["summary"], "Para one.\n\nPara two.")
        # Raw newlines inside a JSON string are technically invalid; be tolerant.
        raw = '{"summary": "Para one.\n\nPara two.", "main_topics": ["a"]}'
        self.assertEqual(_parse_structured_summary(raw)["summary"], "Para one.\n\nPara two.")

    @patch("summarizer.services.summarization._generate")
    @patch("summarizer.services.summarization.AutoTokenizer")
    @patch("summarizer.services.summarization.AutoModelForCausalLM")
    def test_summary_word_count_is_reported(self, mock_model_class, mock_tokenizer_class, mock_generate):
        self._mock_loaded_model(mock_model_class, mock_tokenizer_class)
        mock_generate.return_value = (
            '{"summary": "one two three four five", "main_topics": [], "key_points": [], "conclusions": []}'
        )
        result = summarize_transcript({"chunks": [{"chunk_id": 1, "start": 0.0, "end": 1.0, "text": "tiny"}]})
        self.assertEqual(result["timings"]["summary_words"], 5)

    def test_generation_budgets_are_tight_for_speed(self):
        # Decoding is ~7 tokens/s on the target GPU, so these caps matter.
        self.assertLessEqual(summarization_module.CHUNK_MAX_NEW_TOKENS, 150)
        self.assertGreaterEqual(summarization_module.CHUNK_BATCH_SIZE, 2)

    @patch("summarizer.services.summarization.threading.Thread")
    def test_preload_noop_when_model_already_loaded(self, mock_thread_class):
        summarization_module._model = MagicMock()
        summarization_module.preload_model_async()
        mock_thread_class.assert_not_called()

    @patch("summarizer.services.summarization.threading.Thread")
    def test_preload_starts_background_thread_when_not_loaded(self, mock_thread_class):
        summarization_module._model = None
        summarization_module.preload_model_async()
        mock_thread_class.assert_called_once()
        mock_thread_class.return_value.start.assert_called_once()


class CheckVideoViewIntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()

        # Prevent real background model-loading threads from being spawned
        # while testing the view - preloading is an optimization detail
        # covered by its own tests, and letting it run for real here would
        # race with other tests' model mocks.
        preload_whisper_patcher = patch("summarizer.views.summarize.preload_whisper_async")
        preload_mistral_patcher = patch("summarizer.views.summarize.preload_mistral_async")
        preload_embeddings_patcher = patch("summarizer.views.summarize.preload_embeddings_async")
        index_video_patcher = patch("summarizer.views.summarize.index_video")
        self.mock_preload_whisper = preload_whisper_patcher.start()
        self.mock_preload_mistral = preload_mistral_patcher.start()
        preload_embeddings_patcher.start()
        unload_whisper_patcher = patch("summarizer.views.summarize.unload_whisper")
        self.mock_unload_whisper = unload_whisper_patcher.start()
        self.addCleanup(unload_whisper_patcher.stop)
        cache_patcher = patch("summarizer.views.summarize.get_cached_summary", return_value=None)
        save_patcher = patch("summarizer.views.summarize.save_summary")
        self.mock_get_cached = cache_patcher.start()
        self.mock_save_summary = save_patcher.start()
        self.addCleanup(cache_patcher.stop)
        self.addCleanup(save_patcher.stop)
        self.mock_index_video = index_video_patcher.start()
        self.mock_index_video.return_value = {"success": True, "video": MagicMock(), "reused": False}
        self.addCleanup(preload_whisper_patcher.stop)
        self.addCleanup(preload_mistral_patcher.stop)
        self.addCleanup(preload_embeddings_patcher.stop)
        self.addCleanup(index_video_patcher.stop)

        # Question detection is exercised by its own test suite; here it
        # should just be a well-behaved no-op unless a test overrides it. It's
        # dispatched to a background thread by the view (see
        # detect_and_save_questions_async), so tests assert it was *kicked
        # off* with the right arguments, not on its (async) completion.
        detect_questions_async_patcher = patch(
            "summarizer.views.summarize.detect_and_save_questions_async"
        )
        get_cached_questions_patcher = patch(
            "summarizer.views.summarize.get_cached_questions", return_value=None
        )
        self.mock_detect_questions_async = detect_questions_async_patcher.start()
        self.mock_get_cached_questions = get_cached_questions_patcher.start()
        self.addCleanup(detect_questions_async_patcher.stop)
        self.addCleanup(get_cached_questions_patcher.stop)

        self.accessible_metadata = {
            "accessible": True,
            "title": "Example Video",
            "channel": "Example Channel",
            "duration": "12:35",
            "thumbnail": "https://example.com/thumb.jpg",
        }

        self.successful_audio = {
            "success": True,
            "path": Path("/tmp/testvid.wav"),
            "format": "wav",
            "sample_rate": 16000,
            "channels": 1,
        }
        self.successful_transcript = {
            "success": True,
            "language": "en",
            "duration": 12.8,
            "segments": [{"start": 0.0, "end": 5.2, "text": "Welcome to today's video."}],
            "text": "Welcome to today's video.",
            "device": "cuda",
            "model": "small",
        }

    def test_invalid_url_does_not_call_downstream_services(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe:
            response = self.client.get("/check-video/", {"url": "https://google.com"})
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "url_validation")
        mock_check.assert_not_called()
        mock_extract.assert_not_called()
        mock_transcribe.assert_not_called()
        # An invalid URL should never trigger the (comparatively expensive)
        # background model preload either.
        self.mock_preload_whisper.assert_not_called()
        self.mock_preload_mistral.assert_not_called()

    def test_inaccessible_video_does_not_call_extract_or_transcribe(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe:
            mock_check.return_value = {"accessible": False, "error": "unavailable"}
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "accessibility")
        mock_extract.assert_not_called()
        mock_transcribe.assert_not_called()

    def test_failed_audio_extraction_does_not_call_transcribe(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = {
                "success": False,
                "error": "We couldn't extract audio from this video. Please try another video.",
            }
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "audio_extraction")
        self.assertNotIn("Traceback", data["message"])
        mock_transcribe.assert_not_called()

    def test_full_pipeline_success_returns_transcript_and_summary(self):
        successful_summary = {
            "success": True,
            "chunk_summaries": [
                {"chunk_id": 1, "start": 0.0, "end": 5.2, "summary": "An intro to the video."}
            ],
            "summary": "The video introduces its topic.",
            "main_topics": ["introduction"],
            "key_points": ["sets up the topic"],
            "conclusions": ["viewers get an overview"],
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "device": "cuda:0",
        }
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe, patch(
            "summarizer.views.summarize.summarize_transcript"
        ) as mock_summarize:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = self.successful_audio
            mock_transcribe.return_value = self.successful_transcript
            mock_summarize.return_value = successful_summary
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["title"], "Example Video")
        self.assertEqual(data["data"]["audio"]["format"], "WAV")
        transcript = data["data"]["transcript"]
        self.assertEqual(transcript["language"], "en")
        self.assertEqual(transcript["device"], "cuda")
        self.assertEqual(len(transcript["segments"]), 1)
        self.assertEqual(transcript["full_text"], "Welcome to today's video.")
        self.assertEqual(len(transcript["chunks"]), 1)
        self.assertEqual(transcript["chunks"][0]["chunk_id"], 1)
        mock_transcribe.assert_called_once_with(self.successful_audio["path"])
        # A valid, accessible video should kick off both model preloads early,
        # so their load time overlaps with accessibility/audio-extraction work.
        self.mock_preload_whisper.assert_called_once()
        self.mock_preload_mistral.assert_called_once()
        # Whisper's VRAM is released after transcription, before Mistral runs.
        self.mock_unload_whisper.assert_called_once()
        self.mock_index_video.assert_called_once()
        self.assertEqual(data["data"]["video_id"], VALID_ID)
        self.assertTrue(data["data"]["chat_available"])

        summary = data["data"]["summary"]
        self.assertEqual(summary["summary"], "The video introduces its topic.")
        self.assertEqual(summary["main_topics"], ["introduction"])
        self.assertEqual(summary["key_points"], ["sets up the topic"])
        self.assertEqual(summary["conclusions"], ["viewers get an overview"])
        self.assertEqual(len(summary["chunk_summaries"]), 1)
        mock_summarize.assert_called_once()

    def _run_full_pipeline(self, extra_query=None):
        summary = {
            "success": True, "chunk_summaries": [], "summary": "S", "main_topics": ["t"],
            "key_points": ["k"], "conclusions": ["c"], "model": "m", "device": "cuda:0",
        }
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe, patch(
            "summarizer.views.summarize.summarize_transcript"
        ) as mock_summarize:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = self.successful_audio
            mock_transcribe.return_value = self.successful_transcript
            mock_summarize.return_value = summary
            query = {"url": f"https://www.youtube.com/watch?v={VALID_ID}", **(extra_query or {})}
            response = self.client.get("/check-video/", query)
            return response.json(), (mock_check, mock_extract, mock_transcribe, mock_summarize)

    def test_summary_is_saved_after_a_successful_run(self):
        data, _ = self._run_full_pipeline()
        self.assertTrue(data["success"])
        self.assertFalse(data["data"]["cached"])
        self.mock_save_summary.assert_called_once()
        saved = self.mock_save_summary.call_args.args[1]
        self.assertEqual(saved["summary"], "S")
        self.assertEqual(saved["main_topics"], ["t"])
        self.assertEqual(saved["conclusions"], ["c"])

    def test_degraded_plain_text_summary_is_not_cached(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe, patch(
            "summarizer.views.summarize.summarize_transcript"
        ) as mock_summarize:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = self.successful_audio
            mock_transcribe.return_value = self.successful_transcript
            mock_summarize.return_value = {
                "success": True, "chunk_summaries": [], "summary": "raw text",
                "raw_output": "raw text", "model": "m", "device": "cuda:0",
            }
            data = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            ).json()
        self.assertTrue(data["success"])
        self.mock_save_summary.assert_not_called()

    def test_indexing_receives_the_thumbnail(self):
        self._run_full_pipeline()
        metadata = self.mock_index_video.call_args.args[2]
        self.assertEqual(metadata["thumbnail"], "https://example.com/thumb.jpg")

    def test_question_detection_is_dispatched_to_background_not_awaited(self):
        # Question detection must never block the response: the view returns
        # with an empty, "pending" question list and only *starts* the
        # background job - it must not wait on it (a mock that never
        # completes would hang the test if the view awaited it).
        data, _ = self._run_full_pipeline()

        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["questions"], [])
        self.assertTrue(data["data"]["questions_pending"])
        self.mock_detect_questions_async.assert_called_once()
        call_video, call_processed = self.mock_detect_questions_async.call_args.args
        self.assertEqual(call_video, self.mock_index_video.return_value["video"])
        self.assertIn("segments", call_processed)

    def test_question_detection_is_not_dispatched_when_indexing_failed(self):
        self.mock_index_video.return_value = {"success": False, "video": None, "reused": False}

        data, _ = self._run_full_pipeline()

        self.assertTrue(data["success"])
        self.assertFalse(data["data"]["questions_pending"])
        self.mock_detect_questions_async.assert_not_called()

    def test_already_summarised_video_is_served_from_cache_without_any_processing(self):
        cached_video = MagicMock(
            video_id=VALID_ID, title="Cached title", channel="Cached channel", duration="3:33",
            thumbnail="https://example.com/t.jpg",
            summary={"summary": "Cached summary", "main_topics": ["x"], "key_points": [], "conclusions": []},
        )
        self.mock_get_cached.return_value = cached_video

        data, mocks = self._run_full_pipeline()

        self.assertTrue(data["success"])
        self.assertTrue(data["data"]["cached"])
        self.assertTrue(data["data"]["chat_available"])
        self.assertEqual(data["data"]["title"], "Cached title")
        self.assertEqual(data["data"]["summary"]["summary"], "Cached summary")
        for mock in mocks:
            mock.assert_not_called()
        self.mock_index_video.assert_not_called()
        self.mock_save_summary.assert_not_called()
        self.mock_preload_whisper.assert_not_called()

    def test_cached_response_includes_previously_detected_questions(self):
        cached_video = MagicMock(
            video_id=VALID_ID, title="Cached title", channel="Cached channel", duration="3:33",
            thumbnail="https://example.com/t.jpg",
            summary={"summary": "Cached summary", "main_topics": [], "key_points": [], "conclusions": []},
        )
        self.mock_get_cached.return_value = cached_video
        cached_question = MagicMock(question="What is X?", start=5.0, end=9.0)
        self.mock_get_cached_questions.return_value = [cached_question]

        data, _ = self._run_full_pipeline()

        self.mock_get_cached_questions.assert_called_once_with(cached_video)
        self.assertEqual(len(data["data"]["questions"]), 1)
        self.assertEqual(data["data"]["questions"][0]["question"], "What is X?")
        self.assertFalse(data["data"]["questions_pending"])
        self.mock_detect_questions_async.assert_not_called()

    def test_refresh_parameter_bypasses_the_cache(self):
        self.mock_get_cached.return_value = MagicMock()
        data, mocks = self._run_full_pipeline({"refresh": "1"})
        self.assertFalse(data["data"]["cached"])
        self.mock_get_cached.assert_not_called()
        mocks[3].assert_called_once()

    def test_failed_summarization_returns_failure(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe, patch(
            "summarizer.views.summarize.summarize_transcript"
        ) as mock_summarize:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = self.successful_audio
            mock_transcribe.return_value = self.successful_transcript
            mock_summarize.return_value = {
                "success": False,
                "error": "We couldn't generate a summary for this video. Please try again.",
            }
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "summarization")
        self.assertNotIn("Traceback", data["message"])

    def test_failed_transcription_returns_failure(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = self.successful_audio
            mock_transcribe.return_value = {
                "success": False,
                "error": "We couldn't transcribe the audio. Please try again.",
            }
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "transcription")
        self.assertNotIn("Traceback", data["message"])

    def test_transcript_with_only_empty_segments_fails_processing(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check, patch(
            "summarizer.views.summarize.extract_audio"
        ) as mock_extract, patch(
            "summarizer.views.summarize.transcribe_audio"
        ) as mock_transcribe:
            mock_check.return_value = self.accessible_metadata
            mock_extract.return_value = self.successful_audio
            mock_transcribe.return_value = {
                "success": True,
                "language": "en",
                "duration": 3.0,
                "segments": [{"start": 0.0, "end": 3.0, "text": "   "}],
                "text": "",
                "device": "cuda",
                "model": "small",
            }
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "transcript_processing")
        self.assertEqual(data["message"], "No usable transcript was found.")

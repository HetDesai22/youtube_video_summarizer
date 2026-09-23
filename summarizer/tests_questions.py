"""Tests for automatic question detection (spec Tests 1-7).

Mistral and the embedding model are mocked (loading them per test run would be
slow and GPU-bound); the lightweight candidate detector, dedup logic, and
Django wiring are exercised for real.
"""
from unittest.mock import MagicMock, patch

from django.test import TestCase, TransactionTestCase

from .models import Video, VideoQuestion
from .services.question_detection import (
    MAX_CANDIDATES,
    MAX_QUESTIONS,
    _build_candidate_windows,
    _is_grounded,
    _is_pure_filler,
    _normalize_question,
    detect_and_save_questions_async,
    detect_questions,
    serialize_questions,
)
from .services.video_index import get_cached_questions, save_questions


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def _mock_batch(answers):
    """A _generate_batch replacement that hands out `answers` in order across
    however many batched calls _validate_candidates makes (it may call
    _generate_batch more than once when there are more candidates than
    VALIDATION_BATCH_SIZE), so each candidate gets its intended answer rather
    than every call replaying the same prefix of `answers`.
    """
    position = [0]

    def _side_effect(model, tokenizer, messages_list, max_new_tokens):
        start = position[0]
        end = start + len(messages_list)
        position[0] = end
        return answers[start:end]

    return _side_effect


class CandidateDetectionTests(TestCase):
    """Test 1/2/3/5 at the lightweight-detection layer (no Mistral)."""

    def test_literal_question_mark_is_a_candidate(self):
        segments = [seg(0.0, 4.0, "What is machine learning?"), seg(4.0, 12.0, "It is a method.")]
        windows = _build_candidate_windows(segments)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start"], 0.0)
        self.assertIn("What is machine learning?", windows[0]["text"])

    def test_plain_statement_is_not_a_candidate(self):
        segments = [seg(0.0, 6.0, "Machine learning is a method for learning patterns from data.")]
        self.assertEqual(_build_candidate_windows(segments), [])

    def test_spoken_question_without_question_mark_is_a_candidate(self):
        segments = [
            seg(0.0, 5.0, "Now we have discussed the basics."),
            seg(5.0, 9.0, "So what happens when the model overfits"),
            seg(9.0, 15.0, "In that situation the model performs poorly."),
        ]
        windows = _build_candidate_windows(segments)
        self.assertEqual(len(windows), 1)
        # Anchored to the segment where the question begins, not segment 0.
        self.assertEqual(windows[0]["start"], 5.0)
        self.assertEqual(windows[0]["segment_index"], 1)

    def test_consecutive_candidate_segments_are_merged_into_one_window(self):
        segments = [
            seg(0.0, 4.0, "What is machine learning?"),
            seg(4.0, 8.0, "Why does it matter?"),
            seg(8.0, 14.0, "It matters because it powers many products."),
        ]
        windows = _build_candidate_windows(segments)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start"], 0.0)  # anchored to the first of the run

    def test_pure_filler_sentences_are_not_candidates(self):
        for filler in ["Okay?", "Are you ready?", "Does that make sense?", "Can you hear me?"]:
            self.assertEqual(
                _build_candidate_windows([seg(0.0, 2.0, filler)]), [], f"{filler!r} should be filtered"
            )

    def test_is_pure_filler_does_not_flag_substantive_questions(self):
        self.assertFalse(_is_pure_filler("What is machine learning?"))
        self.assertFalse(_is_pure_filler("Can AI replace programmers?"))


class GroundingFilterTests(TestCase):
    """Real cases captured from a live run where Mistral synthesized a new
    question from the general topic instead of extracting what was actually
    asked - `_is_grounded` is the mechanical backstop that catches this
    regardless of how well the model follows the prompt.
    """

    def test_fabricated_question_with_no_basis_in_excerpt_is_rejected(self):
        self.assertFalse(
            _is_grounded(
                "What is the speed of the video playback?",
                "So we see a lot of output and then there is something that is going on here.",
            )
        )

    def test_fabricated_question_inferring_pros_and_cons_is_rejected(self):
        self.assertFalse(
            _is_grounded(
                "What is the advantage or disadvantage of the slow method compared to the fast method?",
                "Now, so why did this happen?",
            )
        )

    def test_genuine_extraction_from_the_window_is_accepted(self):
        self.assertTrue(
            _is_grounded(
                "What happens when the model overfits?",
                "Now we have discussed the basics. So what happens when the model overfits "
                "In that situation the model performs poorly on new data.",
            )
        )

    def test_lightly_reworded_extraction_is_still_accepted(self):
        self.assertTrue(
            _is_grounded(
                "How do you convert MP4 files to MKV using a bash loop?",
                "let us say you have a movies folder all of them and you want to convert "
                "every single one of them using a bash loop into MKV files",
            )
        )


@patch("summarizer.services.question_detection._load_model", return_value=(MagicMock(), MagicMock()))
class HedgeRejectionTests(TestCase):
    @patch("summarizer.services.question_detection._generate_batch")
    def test_parenthetical_hedge_is_rejected_even_with_grounded_words(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(
            [
                "What is required to install Homebrew? (Note: This is not an exact question "
                "from the excerpt, but it is an implied question based on the context.)"
            ]
        )
        result = detect_questions(
            {"segments": [seg(0.0, 5.0, "If you do not have Homebrew installed, get it.")]}
        )
        self.assertEqual(result["questions"], [])

    @patch("summarizer.services.question_detection._generate_batch")
    def test_assumes_hedge_is_rejected(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(
            ["Why don't we take the faster approach? (assuming the command refers to that one)"]
        )
        result = detect_questions({"segments": [seg(0.0, 5.0, "We removed the first 10 seconds.")]})
        self.assertEqual(result["questions"], [])

    @patch("summarizer.services.question_detection._generate_batch")
    def test_any_parenthetical_aside_is_rejected_even_with_novel_wording(self, mock_batch, mock_load):
        # Real output from a live run: a hedge phrase my regex hadn't seen
        # before ("is not stated in the excerpt") - caught because it's
        # parenthetical at all, not because of its specific wording.
        mock_batch.side_effect = _mock_batch(
            ["What if you want to keep only a portion? (What portion specifically is not "
             "stated in the excerpt)"]
        )
        result = detect_questions(
            {"segments": [seg(0.0, 5.0, "In the second case, we remove the last 10 seconds.")]}
        )
        self.assertEqual(result["questions"], [])


class NormalizeQuestionTests(TestCase):
    def test_adds_question_mark_and_capitalizes(self):
        self.assertEqual(_normalize_question("what is overfitting"), "What is overfitting?")

    def test_strips_quotes_and_answer_label(self):
        self.assertEqual(_normalize_question('Answer: "What is overfitting?"'), "What is overfitting?")

    def test_strips_question_label(self):
        self.assertEqual(_normalize_question("Question: Can AI replace programmers?"), "Can AI replace programmers?")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(_normalize_question("what   is\n\noverfitting"), "What is overfitting?")


@patch("summarizer.services.question_detection._load_model", return_value=(MagicMock(), MagicMock()))
class DetectQuestionsTests(TestCase):
    def test_no_segments_returns_empty_without_calling_mistral(self, mock_load):
        result = detect_questions({"segments": []})
        self.assertTrue(result["success"])
        self.assertEqual(result["questions"], [])
        mock_load.assert_not_called()

    def test_no_candidates_returns_empty_without_calling_mistral(self, mock_load):
        result = detect_questions({"segments": [seg(0.0, 5.0, "This is just a statement.")]})
        self.assertTrue(result["success"])
        self.assertEqual(result["questions"], [])
        mock_load.assert_not_called()

    @patch("summarizer.services.question_detection._generate_batch")
    def test_explicit_question_is_detected_with_correct_timestamp(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(["What is machine learning?"])
        result = detect_questions(
            {"segments": [seg(0.0, 4.0, "What is machine learning?"), seg(4.0, 12.0, "It's a method.")]}
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(result["questions"]), 1)
        q = result["questions"][0]
        self.assertEqual(q["question"], "What is machine learning?")
        self.assertEqual(q["start"], 0.0)

    @patch("summarizer.services.question_detection._generate_batch")
    def test_model_declining_with_none_produces_no_question(self, mock_batch, mock_load):
        # A candidate slips through lightweight detection (e.g. "Is" mid-clause)
        # but Mistral correctly judges there's no real question.
        mock_batch.side_effect = _mock_batch(["NONE"])
        result = detect_questions({"segments": [seg(0.0, 5.0, "Is this the best approach.")]})
        self.assertTrue(result["success"])
        self.assertEqual(result["questions"], [])

    @patch("summarizer.services.question_detection._generate_batch")
    def test_none_with_trailing_commentary_is_still_treated_as_none(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(["None (this is a statement, not a question.)"])
        result = detect_questions({"segments": [seg(0.0, 5.0, "Is this the best approach.")]})
        self.assertEqual(result["questions"], [])

    @patch("summarizer.services.question_detection._generate_batch")
    def test_duplicate_questions_are_deduplicated_keeping_earliest(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(
            ["What is machine learning?", "What exactly is machine learning?"]
        )
        result = detect_questions(
            {
                "segments": [
                    seg(0.0, 4.0, "Let's first ask, what is machine learning?"),
                    seg(30.0, 34.0, "So what exactly is machine learning anyway?"),
                    seg(60.0, 65.0, "It is a method for learning from data."),
                ]
            }
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["start"], 0.0)

    @patch("summarizer.services.question_detection._generate_batch")
    def test_genuinely_different_questions_are_both_kept(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(
            ["What is machine learning?", "Why is training data important?"]
        )
        result = detect_questions(
            {
                "segments": [
                    seg(0.0, 4.0, "What is machine learning?"),
                    seg(300.0, 305.0, "Why is training data important?"),
                ]
            }
        )
        self.assertEqual(len(result["questions"]), 2)

    @patch("summarizer.services.question_detection._generate_batch")
    def test_results_are_ordered_chronologically_regardless_of_segment_order(
        self, mock_batch, mock_load
    ):
        # Candidates are naturally in transcript order here, but the dedup/rank
        # step re-sorts explicitly - assert the final invariant holds.
        mock_batch.side_effect = _mock_batch(["Question A?", "Question B?", "Question C?"])
        result = detect_questions(
            {
                "segments": [
                    seg(0.0, 4.0, "Question A?"),
                    seg(100.0, 104.0, "Question B?"),
                    seg(200.0, 204.0, "Question C?"),
                ]
            }
        )
        starts = [q["start"] for q in result["questions"]]
        self.assertEqual(starts, sorted(starts))

    @patch("summarizer.services.question_detection._generate_batch")
    def test_more_than_max_questions_are_capped(self, mock_batch, mock_load):
        # Lexically distinct subjects, so the real embedding model doesn't
        # dedupe them away before the cap logic under test even runs.
        subjects = [
            "machine learning", "overfitting", "gradient descent", "neural networks",
            "training data", "backpropagation", "regularization", "the loss function",
            "hyperparameters", "convolutional layers", "transfer learning", "dropout",
            "batch normalization", "the learning rate", "cross validation", "ensembling",
            "feature engineering",
        ]
        self.assertGreater(len(subjects), MAX_QUESTIONS)
        answers = [f"What is {subject}?" for subject in subjects]
        mock_batch.side_effect = _mock_batch(answers)
        segments = [
            seg(i * 30.0, i * 30.0 + 4.0, answer) for i, answer in enumerate(answers)
        ]
        result = detect_questions({"segments": segments})
        self.assertEqual(len(result["questions"]), MAX_QUESTIONS)

    @patch("summarizer.services.question_detection._generate_batch")
    def test_more_than_max_candidates_are_pre_filtered_before_mistral(self, mock_batch, mock_load):
        mock_batch.side_effect = _mock_batch(["NONE"] * MAX_CANDIDATES)
        segments = [
            seg(i * 10.0, i * 10.0 + 4.0, f"What is topic number {i}?")
            for i in range(MAX_CANDIDATES + 20)
        ]
        detect_questions({"segments": segments})
        total_sent = sum(len(call.args[2]) for call in mock_batch.call_args_list)
        self.assertEqual(total_sent, MAX_CANDIDATES)

    @patch("summarizer.services.question_detection._generate_batch")
    def test_oom_returns_clean_error(self, mock_batch, mock_load):
        import torch

        mock_batch.side_effect = torch.OutOfMemoryError("CUDA OOM")
        result = detect_questions({"segments": [seg(0.0, 4.0, "What is machine learning?")]})
        self.assertFalse(result["success"])
        self.assertNotIn("CUDA", result["error"])

    @patch("summarizer.services.question_detection._generate_batch")
    def test_unexpected_exception_returns_clean_error(self, mock_batch, mock_load):
        mock_batch.side_effect = RuntimeError("boom")
        result = detect_questions({"segments": [seg(0.0, 4.0, "What is machine learning?")]})
        self.assertFalse(result["success"])
        self.assertNotIn("boom", result["error"])


class DetectAndSaveQuestionsAsyncTests(TransactionTestCase):
    """The view dispatches detection to a background thread so it doesn't
    block the summarize response; these tests join() the returned thread to
    wait for completion (fast here since Mistral is mocked) rather than
    guessing at timing.

    Uses TransactionTestCase (not TestCase): the background thread opens its
    own database connection, which cannot see rows from the main thread's
    still-open, uncommitted TestCase transaction. TransactionTestCase commits
    for real (and cleans up by truncating tables instead), so the video
    created in setUp is actually visible to that other connection.
    """

    def setUp(self):
        self.video = Video.objects.create(video_id="abcDEFghijk", url="https://youtu.be/abcDEFghijk")

    @patch("summarizer.services.question_detection.detect_questions")
    def test_successful_detection_is_saved_in_the_background(self, mock_detect):
        mock_detect.return_value = {
            "success": True,
            "questions": [{"question": "What is X?", "start": 5.0, "end": 9.0}],
        }
        detect_and_save_questions_async(self.video, {"segments": []}).join(timeout=2)

        cached = get_cached_questions(self.video)
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0].question, "What is X?")

    @patch("summarizer.services.question_detection.detect_questions")
    def test_failed_detection_saves_nothing_and_does_not_raise(self, mock_detect):
        mock_detect.return_value = {"success": False, "error": "boom"}
        detect_and_save_questions_async(self.video, {"segments": []}).join(timeout=2)

        self.assertIsNone(get_cached_questions(self.video))

    @patch("summarizer.services.question_detection.detect_questions")
    def test_crash_in_the_background_thread_does_not_propagate(self, mock_detect):
        mock_detect.side_effect = RuntimeError("boom")
        # Must not raise in the calling (main) thread - the crash happens
        # entirely inside the background thread.
        detect_and_save_questions_async(self.video, {"segments": []}).join(timeout=2)
        self.assertIsNone(get_cached_questions(self.video))


class SerializeQuestionsTests(TestCase):
    def test_serializes_plain_dicts_with_real_timestamp_and_url(self):
        questions = [{"question": "What is ML?", "start": 134.2, "end": 142.7, "segment_index": 3}]
        result = serialize_questions("abcDEFghijk", questions)
        self.assertEqual(
            result,
            [
                {
                    "id": 1,
                    "question": "What is ML?",
                    "start": 134.2,
                    "end": 142.7,
                    "timestamp": "02:14",
                    "youtube_url": "https://www.youtube.com/watch?v=abcDEFghijk&t=134s",
                }
            ],
        )

    def test_serializes_model_instances_from_the_cache(self):
        video = Video.objects.create(video_id="abcDEFghijk", url="https://youtu.be/abcDEFghijk")
        VideoQuestion.objects.create(video=video, question="Why does it matter?", start=337.4, end=349.1)
        result = serialize_questions(video.video_id, video.questions.all())
        self.assertEqual(result[0]["question"], "Why does it matter?")
        self.assertEqual(result[0]["timestamp"], "05:37")

    def test_ids_are_sequential_starting_at_one(self):
        questions = [
            {"question": "A?", "start": 0.0, "end": 1.0},
            {"question": "B?", "start": 10.0, "end": 11.0},
        ]
        result = serialize_questions("abcDEFghijk", questions)
        self.assertEqual([q["id"] for q in result], [1, 2])


class QuestionCacheTests(TestCase):
    def setUp(self):
        self.video = Video.objects.create(video_id="abcDEFghijk", url="https://youtu.be/abcDEFghijk")

    def test_no_saved_questions_is_not_a_cache_hit(self):
        self.assertIsNone(get_cached_questions(self.video))

    def test_saved_questions_are_returned_in_order(self):
        save_questions(
            self.video,
            [
                {"question": "B?", "start": 10.0, "end": 11.0, "segment_index": 1},
                {"question": "A?", "start": 0.0, "end": 1.0, "segment_index": 0},
            ],
        )
        cached = get_cached_questions(self.video)
        self.assertEqual([q.question for q in cached], ["A?", "B?"])

    def test_resaving_replaces_old_questions_not_appends(self):
        save_questions(self.video, [{"question": "Old?", "start": 0.0, "end": 1.0}])
        save_questions(self.video, [{"question": "New?", "start": 0.0, "end": 1.0}])
        cached = get_cached_questions(self.video)
        self.assertEqual([q.question for q in cached], ["New?"])

    def test_questions_are_isolated_per_video(self):
        other = Video.objects.create(video_id="videoBBBBBBB", url="https://youtu.be/videoBBBBBBB")
        save_questions(self.video, [{"question": "For A?", "start": 0.0, "end": 1.0}])
        save_questions(other, [{"question": "For B?", "start": 0.0, "end": 1.0}])
        self.assertEqual([q.question for q in get_cached_questions(self.video)], ["For A?"])
        self.assertEqual([q.question for q in get_cached_questions(other)], ["For B?"])

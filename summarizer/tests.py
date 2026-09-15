from unittest.mock import patch

import yt_dlp
from django.test import Client, TestCase

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


class CheckVideoViewIntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_invalid_url_does_not_call_yt_dlp(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check:
            response = self.client.get("/check-video/", {"url": "https://google.com"})
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "url_validation")
        mock_check.assert_not_called()

    def test_valid_url_accessible_video_returns_success(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check:
            mock_check.return_value = {
                "accessible": True,
                "title": "Example Video",
                "channel": "Example Channel",
                "duration": "12:35",
                "thumbnail": "https://example.com/thumb.jpg",
            }
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["title"], "Example Video")
        mock_check.assert_called_once()

    def test_valid_url_inaccessible_video_returns_failure(self):
        with patch("summarizer.views.summarize.check_video_accessibility") as mock_check:
            mock_check.return_value = {"accessible": False, "error": "unavailable"}
            response = self.client.get(
                "/check-video/", {"url": f"https://www.youtube.com/watch?v={VALID_ID}"}
            )
            data = response.json()

        self.assertFalse(data["success"])
        self.assertEqual(data["stage"], "accessibility")

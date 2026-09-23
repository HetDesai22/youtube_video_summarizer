import logging
import subprocess
from pathlib import Path

import yt_dlp
from django.conf import settings

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(settings.BASE_DIR) / "media" / "audio"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

_GENERIC_ERROR = "We couldn't extract audio from this video. Please try another video."


class _SilentLogger:
    """Swallows yt-dlp's own console output; failures are logged by us instead."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def _cleanup(*paths):
    for path in paths:
        try:
            if path and path.exists():
                path.unlink()
        except OSError:
            logger.warning("Could not remove temporary file: %s", path)


def extract_audio(url, video_id):
    """Download a YouTube video's audio track and standardize it to a 16kHz
    mono WAV file.

    `url` must already be a validated, accessible YouTube URL; this function
    performs no validation of its own. Returns
    {"success": True, "path": Path, "format": "wav", "sample_rate": 16000,
    "channels": 1} on success, or {"success": False, "error": str} on
    failure. Technical exceptions are logged, never returned to the caller.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    final_path = AUDIO_DIR / f"{video_id}.wav"
    temp_template = AUDIO_DIR / f"{video_id}.source.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(temp_template),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "logger": _SilentLogger(),
    }

    downloaded_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))
    except Exception:
        logger.exception("yt-dlp failed to download audio for video %s", video_id)
        downloaded_path = None

    if not downloaded_path or not downloaded_path.exists():
        matches = list(AUDIO_DIR.glob(f"{video_id}.source.*"))
        downloaded_path = matches[0] if matches else None

    if not downloaded_path or not downloaded_path.exists():
        logger.error("No downloaded audio file found for video %s", video_id)
        return {"success": False, "error": _GENERIC_ERROR}

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(downloaded_path),
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", str(TARGET_CHANNELS),
                "-vn",
                str(final_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception:
        logger.exception("FFmpeg conversion crashed for video %s", video_id)
        _cleanup(downloaded_path, final_path)
        return {"success": False, "error": _GENERIC_ERROR}

    _cleanup(downloaded_path)

    if result.returncode != 0:
        logger.error("FFmpeg failed for video %s: %s", video_id, result.stderr[-2000:])
        _cleanup(final_path)
        return {"success": False, "error": _GENERIC_ERROR}

    if not final_path.exists() or final_path.stat().st_size == 0:
        logger.error("FFmpeg produced an empty or missing file for video %s", video_id)
        _cleanup(final_path)
        return {"success": False, "error": _GENERIC_ERROR}

    return {
        "success": True,
        "path": final_path,
        "format": "wav",
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": TARGET_CHANNELS,
    }

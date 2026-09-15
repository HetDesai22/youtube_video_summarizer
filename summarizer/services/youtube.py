import logging

import yt_dlp

logger = logging.getLogger(__name__)

_INACCESSIBLE_MESSAGE = "The YouTube video is unavailable or cannot be accessed."


class _SilentLogger:
    """Swallows yt-dlp's own console output; failures are logged by us instead."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "noplaylist": True,
    "socket_timeout": 15,
    "logger": _SilentLogger(),
}


def _format_duration(seconds):
    if not seconds:
        return "Unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def check_video_accessibility(url):
    """Check whether a YouTube video can be accessed via yt-dlp, without
    downloading anything.

    Returns {"accessible": True, "title": ..., "channel": ..., "duration": ...,
    "thumbnail": ...} on success, or {"accessible": False, "error": str} on
    failure. Technical exceptions are logged, never returned to the caller.
    """
    try:
        with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        logger.warning("yt-dlp could not access video %s: %s", url, exc)
        return {"accessible": False, "error": _INACCESSIBLE_MESSAGE}
    except Exception:
        logger.exception("Unexpected error checking accessibility for %s", url)
        return {"accessible": False, "error": _INACCESSIBLE_MESSAGE}

    if not info:
        return {"accessible": False, "error": _INACCESSIBLE_MESSAGE}

    return {
        "accessible": True,
        "title": info.get("title") or "Unknown title",
        "channel": info.get("channel") or info.get("uploader") or "Unknown channel",
        "duration": _format_duration(info.get("duration")),
        "thumbnail": info.get("thumbnail"),
    }

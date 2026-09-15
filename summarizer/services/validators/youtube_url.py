import re
from urllib.parse import parse_qs, urlparse

_ERROR_MESSAGE = "Please enter a valid YouTube video URL."

_STANDARD_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com"}
_SHORT_HOST = "youtu.be"

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _invalid():
    return {"valid": False, "error": _ERROR_MESSAGE}


def _normalize_host(netloc):
    host = netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def validate_youtube_url(url):
    """Validate a YouTube video URL without making any network requests.

    Returns {"valid": True, "video_id": str, "url": str} on success, or
    {"valid": False, "error": str} on failure.
    """
    if not url or not isinstance(url, str):
        return _invalid()

    url = url.strip()
    if not url:
        return _invalid()

    try:
        parsed = urlparse(url)
    except ValueError:
        return _invalid()

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return _invalid()

    host = _normalize_host(parsed.netloc)
    path = parsed.path.rstrip("/")

    video_id = None

    if host == _SHORT_HOST:
        segment = parsed.path.lstrip("/").split("/")[0]
        video_id = segment or None
    elif host in _STANDARD_HOSTS:
        if path == "/watch":
            values = parse_qs(parsed.query).get("v")
            video_id = values[0] if values else None
        elif path.startswith("/shorts/"):
            video_id = path[len("/shorts/"):].split("/")[0] or None
        else:
            return _invalid()
    else:
        return _invalid()

    if not video_id or not _VIDEO_ID_RE.match(video_id):
        return _invalid()

    return {"valid": True, "video_id": video_id, "url": url}

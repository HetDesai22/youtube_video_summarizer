import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from ..services.video_chat import chat_with_video, get_or_create_session
from ..services.video_index import get_indexed_video

logger = logging.getLogger(__name__)

_MAX_VIDEO_ID_LENGTH = 20


def _error(message, status):
    return JsonResponse({"success": False, "error": message}, status=status)


@require_POST
def chat(request):
    try:
        payload = json.loads(request.body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error("Invalid request.", 400)

    if not isinstance(payload, dict):
        return _error("Invalid request.", 400)

    video_id = str(payload.get("video_id") or "")[:_MAX_VIDEO_ID_LENGTH]
    session_id = payload.get("session_id") or None
    message = payload.get("message")

    if not video_id or not isinstance(message, str) or not message.strip():
        return _error("Please enter a question.", 400)

    video = get_indexed_video(video_id)
    if video is None:
        return _error("This video hasn't been processed yet. Please summarize it first.", 404)

    session = get_or_create_session(video, session_id)
    result = chat_with_video(session, message)
    if not result["success"]:
        return _error(result["error"], 500)

    return JsonResponse(result)

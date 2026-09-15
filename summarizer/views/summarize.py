from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ..services.validators.youtube_url import validate_youtube_url
from ..services.youtube import check_video_accessibility


@require_GET
def check_video(request):
    url = request.GET.get("url", "")

    validation = validate_youtube_url(url)
    if not validation["valid"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "url_validation",
                "message": validation["error"],
            }
        )

    accessibility = check_video_accessibility(validation["url"])
    if not accessibility["accessible"]:
        return JsonResponse(
            {
                "success": False,
                "stage": "accessibility",
                "message": (
                    "The YouTube video is unavailable or cannot be accessed. "
                    "Please check the URL or make sure the video is publicly available."
                ),
            }
        )

    return JsonResponse(
        {
            "success": True,
            "message": "Video is valid and accessible.",
            "data": {
                "title": accessibility["title"],
                "channel": accessibility["channel"],
                "duration": accessibility["duration"],
                "thumbnail": accessibility["thumbnail"],
            },
        }
    )

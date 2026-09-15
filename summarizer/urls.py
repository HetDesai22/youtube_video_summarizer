from django.urls import path

from .views.home import home
from .views.summarize import check_video

urlpatterns = [
    path("", home, name="home"),
    path("check-video/", check_video, name="check-video"),
]

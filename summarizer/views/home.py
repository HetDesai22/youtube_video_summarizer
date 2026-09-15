from django.shortcuts import render


def home(request):
    return render(request, "summarizer/index.html")
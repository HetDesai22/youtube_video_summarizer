import os

from django.apps import AppConfig


class SummarizerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'summarizer'

    def ready(self):
        # Warm up the heavy models as soon as the server actually starts
        # serving, rather than paying their full load cost on the first
        # user request. `RUN_MAIN` is only set in the real worker process
        # that runserver's autoreloader spawns - the initial watcher
        # process would otherwise trigger this twice.
        if os.environ.get("RUN_MAIN") != "true":
            return

        from .services.embeddings import preload_model_async as preload_embeddings_async
        from .services.summarization import preload_model_async as preload_mistral_async

        # Whisper is deliberately NOT preloaded here: it is unloaded after each
        # transcription to keep VRAM free for Mistral, and is preloaded per
        # request instead (overlapping the audio download).
        preload_mistral_async()
        preload_embeddings_async()

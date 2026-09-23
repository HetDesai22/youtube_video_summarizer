import uuid

from django.db import models
from pgvector.django import VectorField

# Must match the embedding model used in services/embeddings.py
# (sentence-transformers/all-MiniLM-L6-v2 produces 384-dimensional vectors).
EMBEDDING_DIMENSIONS = 384


class Video(models.Model):
    class Status(models.TextChoices):
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    video_id = models.CharField(max_length=20, unique=True)
    url = models.URLField(max_length=500)
    title = models.CharField(max_length=500, blank=True)
    channel = models.CharField(max_length=300, blank=True)
    duration = models.CharField(max_length=20, blank=True)
    thumbnail = models.URLField(max_length=500, blank=True)
    # Final summary payload ({summary, main_topics, key_points, conclusions, ...}),
    # so re-submitting an already-processed video is instant.
    summary = models.JSONField(null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PROCESSING
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.video_id} ({self.status})"


class Transcript(models.Model):
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name="transcript")
    language = models.CharField(max_length=10, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    full_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transcript for {self.video.video_id}"


class TranscriptChunk(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="chunks")
    chunk_id = models.PositiveIntegerField()
    start = models.FloatField()
    end = models.FloatField()
    text = models.TextField()
    word_count = models.PositiveIntegerField()
    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS)

    class Meta:
        ordering = ["video", "chunk_id"]
        constraints = [
            models.UniqueConstraint(fields=["video", "chunk_id"], name="unique_chunk_per_video"),
        ]

    def __str__(self):
        return f"{self.video.video_id} chunk {self.chunk_id}"


class VideoQuestion(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="questions")
    question = models.TextField()
    start = models.FloatField()
    end = models.FloatField()
    # Index into the transcript's segment list at detection time (segments
    # aren't stored as their own DB rows) - kept for debugging provenance only,
    # not used by the API.
    source_segment_index = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["video", "start"]

    def __str__(self):
        return f"{self.video.video_id} @ {self.start:.1f}s: {self.question[:40]}"


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="chat_sessions")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.id} on {self.video.video_id}"


class ChatMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=Role.choices)
    message = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self):
        return f"{self.role}: {self.message[:40]}"

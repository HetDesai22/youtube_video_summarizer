"""Benchmark faster-whisper settings on a slice of an already-extracted WAV.

    venv\\Scripts\\python.exe scripts\\bench_whisper.py media\\audio\\VIDEO_ID.wav [seconds]

Reports wall time, GPU memory in use, and word-level similarity to the
sequential beam_size=5 baseline, so a speedup can be weighed against any
accuracy loss.
"""
import difflib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

wav = sys.argv[1]
seconds = sys.argv[2] if len(sys.argv) > 2 else "600"
slice_path = Path(tempfile.gettempdir()) / "bench_slice.wav"
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-t", seconds, str(slice_path)], check=True
)


def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def sequential(model, beam):
    segments, _ = model.transcribe(str(slice_path), vad_filter=False, temperature=0.0, beam_size=beam)
    return " ".join(s.text.strip() for s in segments)


def fixed_windows(duration, window=30.0):
    """Consecutive [{"start": s, "end": e}, ...] windows covering the audio."""
    clips, start = [], 0.0
    while start < duration:
        end = min(start + window, duration)
        clips.append({"start": start, "end": end})
        start = end
    return clips


def batched(model, batch_size):
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(slice_path)],
        capture_output=True, text=True,
    ).stdout.strip())
    pipe = BatchedInferencePipeline(model=model)
    segments, _ = pipe.transcribe(
        str(slice_path), batch_size=batch_size, beam_size=1, vad_filter=False,
        clip_timestamps=fixed_windows(duration),
        without_timestamps=False, temperature=0.0, condition_on_previous_text=False,
    )
    segs = list(segments)
    return " ".join(s.text.strip() for s in segs), len(segs)


model = WhisperModel("small", device="cuda", compute_type="float16")
print(f"GPU used with model loaded (incl. anything else running): {gpu_used_mib()} MiB")

baseline_words = None
runs = [
    ("sequential beam=1 (baseline)", lambda: (sequential(model, 1), None)),
    ("batched bs=8  beam=1", lambda: batched(model, 8)),
    ("batched bs=16 beam=1", lambda: batched(model, 16)),
]
for label, run in runs:
    t0 = time.time()
    text, n_segments = run()
    elapsed = time.time() - t0
    words = text.lower().split()
    if baseline_words is None:
        baseline_words, similarity = words, 1.0
    else:
        similarity = difflib.SequenceMatcher(None, baseline_words, words, autojunk=False).ratio()
    print(
        f"{label:<30} {elapsed:6.1f}s  words={len(words):5d}  segments={n_segments}  "
        f"similarity={similarity:.3f}  gpu={gpu_used_mib()}MiB"
    )

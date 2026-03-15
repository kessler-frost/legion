import subprocess
import threading
import time
from pathlib import Path

RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"

_process: subprocess.Popen = None
_recording = False


def is_recording() -> bool:
    return _recording and _process is not None and _process.poll() is None


def start_recording(video_device: int = 0, audio_device: int = 0) -> Path:
    """Start recording raw video + audio from webcam via ffmpeg."""
    global _process, _recording

    RECORDINGS_DIR.mkdir(exist_ok=True)
    filename = f"rec_{int(time.time())}.mp4"
    filepath = RECORDINGS_DIR / filename

    _process = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "1280x720",
            "-i", f"{video_device}:{audio_device}",
            "-vf", "hflip,vflip",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            str(filepath),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _recording = True
    return filepath


def stop_recording() -> None:
    """Stop recording gracefully."""
    global _process, _recording
    if _process and _process.poll() is None:
        _process.stdin.write(b"q")
        _process.stdin.flush()
        _process.wait(timeout=5)
    _process = None
    _recording = False


def delete_recordings(filenames: list[str]) -> list[str]:
    """Delete recordings by filename."""
    deleted = []
    for name in filenames:
        path = RECORDINGS_DIR / name
        if path.exists() and path.parent == RECORDINGS_DIR:
            path.unlink()
            deleted.append(name)
    return deleted


def list_recordings() -> list[dict]:
    """List all recordings with metadata."""
    RECORDINGS_DIR.mkdir(exist_ok=True)
    recordings = []
    for f in sorted(RECORDINGS_DIR.glob("*.mp4"), reverse=True):
        recordings.append({
            "filename": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            "created": f.stat().st_mtime,
        })
    return recordings

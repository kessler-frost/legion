import subprocess
import time
from pathlib import Path

RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"

_process: subprocess.Popen = None
_recording = False


def is_recording() -> bool:
    return _recording and _process is not None and _process.poll() is None


def start_recording() -> Path:
    """Start screen recording with audio via macOS screencapture."""
    global _process, _recording

    RECORDINGS_DIR.mkdir(exist_ok=True)
    filename = f"rec_{int(time.time())}.mov"
    filepath = RECORDINGS_DIR / filename

    _process = subprocess.Popen(
        [
            "screencapture", "-v", "-g",
            "-D1",
            str(filepath),
        ],
        stdin=subprocess.PIPE,
    )
    _recording = True
    return filepath


def stop_recording() -> None:
    """Stop recording by sending Ctrl+C."""
    global _process, _recording
    if _process and _process.poll() is None:
        _process.terminate()
        _process.wait(timeout=5)
    _process = None
    _recording = False


def delete_recordings(filenames: list[str]) -> list[str]:
    deleted = []
    for name in filenames:
        path = RECORDINGS_DIR / name
        if path.exists() and path.parent == RECORDINGS_DIR:
            path.unlink()
            deleted.append(name)
    return deleted


def list_recordings() -> list[dict]:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    recordings = []
    for f in sorted(RECORDINGS_DIR.glob("*.mov"), reverse=True):
        recordings.append({
            "filename": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            "created": f.stat().st_mtime,
        })
    return recordings

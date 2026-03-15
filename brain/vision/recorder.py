import subprocess
import time
from pathlib import Path

RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"

_process: subprocess.Popen = None
_recording = False
_current_path: Path = None


def is_recording() -> bool:
    return _recording and _process is not None and _process.poll() is None


def start_recording() -> Path:
    """Start screen recording with audio via ffmpeg."""
    global _process, _recording, _current_path

    RECORDINGS_DIR.mkdir(exist_ok=True)
    filename = f"rec_{int(time.time())}.mp4"
    filepath = RECORDINGS_DIR / filename
    _current_path = filepath

    _process = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-capture_cursor", "1",
            "-i", "1:0",  # screen 1 : audio device 0 (C270 mic)
            "-r", "30",
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
        _process.wait(timeout=10)
    _process = None
    _recording = False


def speed_up_recording(filename: str, speed: float = 2.0) -> str:
    """Create a sped-up version of a recording."""
    src = RECORDINGS_DIR / filename
    out_name = f"{src.stem}_{speed}x.mp4"
    out_path = RECORDINGS_DIR / out_name

    # Video speed + audio speed
    vf = f"setpts={1/speed}*PTS"
    af = f"atempo={speed}" if speed <= 2.0 else f"atempo=2.0,atempo={speed/2.0}"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            str(out_path),
        ],
        capture_output=True,
    )
    return out_name


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
    for f in sorted(RECORDINGS_DIR.glob("*.mp4"), reverse=True):
        recordings.append({
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            "created": f.stat().st_mtime,
        })
    return recordings

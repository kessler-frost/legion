import time
import tempfile
import subprocess
import numpy as np
import wave
from pathlib import Path

from mlx_qwen3_asr import transcribe

CHUNK_DURATION = 3  # seconds per audio chunk
SILENCE_THRESHOLD = 0.01  # RMS below this = silence


def extract_audio_chunk(duration: float, output_path: Path):
    """Extract a chunk of audio from iPhone mic to WAV file."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-i", ":0",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-t", str(duration),
            str(output_path),
        ],
        capture_output=True,
    )


def is_silent(wav_path: Path) -> bool:
    """Check if audio chunk is mostly silence."""
    with wave.open(str(wav_path), "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    rms = np.sqrt(np.mean(data.astype(float) ** 2)) / 32768
    return rms < SILENCE_THRESHOLD


def run():
    print("Listening — capturing audio from iPhone mic")
    print(f"Chunk duration: {CHUNK_DURATION}s")

    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        extract_audio_chunk(CHUNK_DURATION, wav_path)

        if wav_path.exists() and wav_path.stat().st_size > 1000 and not is_silent(wav_path):
            result = transcribe(str(wav_path))
            text = result.get("text", "").strip()
            if text:
                print(f"HEARD: {text}")

        wav_path.unlink(missing_ok=True)

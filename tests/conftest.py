"""Shared fixtures.

The API tests run against a fake transcriber, so the suite needs neither a GPU
nor a 1.6 GB model download. Audio conversion is *not* faked — that step runs
real ffmpeg against a generated tone, because that is where format bugs live.
"""

from __future__ import annotations

import math
import struct
import time
import wave
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.transcriber import Segment, TranscriptionInfo

FAKE_SEGMENTS = [
    Segment(index=1, start=0.0, end=1.4, text="مرحبا بكم في هذا التسجيل"),
    Segment(index=2, start=1.4, end=3.2, text="نتحدث اليوم عن تحويل الصوت إلى نص"),
    Segment(index=3, start=3.2, end=4.9, text="والنتيجة تظهر من اليمين إلى اليسار"),
]


class FakeTranscriber:
    """Stands in for :class:`app.transcriber.Transcriber`."""

    def __init__(
        self,
        segments: list[Segment] | None = None,
        *,
        error: Exception | None = None,
        prepare_error: Exception | None = None,
        duration: float = 4.9,
    ) -> None:
        self.segments = FAKE_SEGMENTS if segments is None else segments
        self.error = error
        self.prepare_error = prepare_error
        self.duration = duration
        self.calls: list[dict] = []
        self.prepared: list[str] = []

    def prepare(self, model: str) -> None:
        if self.prepare_error is not None:
            raise self.prepare_error
        self.prepared.append(model)

    def transcribe(
        self, audio_path: Path, *, model: str, language: str, task: str
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        self.calls.append(
            {"path": audio_path, "model": model, "language": language, "task": task}
        )
        if self.error is not None:
            raise self.error
        info = TranscriptionInfo(
            language="ar" if language == "auto" else language,
            language_probability=0.98,
            duration=self.duration,
            model=model,
            device="cpu",
            compute_type="int8",
        )
        return info, iter(self.segments)

    def runtime_info(self) -> dict:
        return {
            "device": "cpu",
            "cuda_devices": 0,
            "loaded_models": [],
            "warnings": [],
        }


def write_tone(path: Path, *, seconds: float = 1.0, rate: int = 22_050) -> Path:
    """Write a mono 16-bit WAV sine tone that ffmpeg can genuinely decode."""
    frames = bytearray()
    total = int(seconds * rate)
    for n in range(total):
        value = int(18_000 * math.sin(2 * math.pi * 440 * n / rate))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return path


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    return write_tone(tmp_path / "tone.wav")


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    """Build a TestClient over a freshly configured app."""

    def factory(transcriber=None, **env: object) -> TestClient:
        monkeypatch.setenv("S2T_WORK_DIR", str(tmp_path / "work"))
        # The developer's own .env must not decide how the suite runs: the gate
        # is off unless a test asks for it via S2T_PASSWORD.
        monkeypatch.setenv("S2T_PASSWORD", "")
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        get_settings.cache_clear()
        app = create_app(transcriber or FakeTranscriber())
        return TestClient(app)

    yield factory
    get_settings.cache_clear()


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:
    with make_client() as test_client:
        yield test_client


def post_audio(client: TestClient, path: Path, **fields: str):
    """Upload a file to /api/transcribe with sensible defaults."""
    data = {"model": "small", "language": "ar", "task": "transcribe"}
    data.update(fields)
    with path.open("rb") as handle:
        return client.post(
            "/api/transcribe",
            files={"file": (path.name, handle, "audio/wav")},
            data=data,
        )


def wait_for(client: TestClient, job_id: str, *statuses: str, timeout: float = 20.0) -> dict:
    """Poll a job until it reaches one of ``statuses``."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        if response.status_code == 404:
            raise AssertionError(f"job {job_id} disappeared")
        last = response.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job stuck in {last.get('status')!r}, wanted {statuses}")

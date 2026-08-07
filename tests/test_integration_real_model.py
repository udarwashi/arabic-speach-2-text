"""Opt-in check against a real Whisper model and real Arabic audio.

Skipped unless you ask for it, because it downloads a model and takes minutes:

    S2T_RUN_SLOW=1 .venv/bin/pytest tests/test_integration_real_model.py -s

Point ``S2T_TEST_AUDIO`` at your own file to use something other than the default.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from app.config import get_settings
from app.transcriber import Transcriber
from tests.conftest import post_audio, wait_for

pytestmark = pytest.mark.slow

DEFAULT_AUDIO = Path("/home/usama/projects/speachToText-arabic/gg.mp3")
ARABIC_LETTERS = re.compile(r"[؀-ۿ]")


def _audio_path() -> Path:
    candidate = Path(os.environ.get("S2T_TEST_AUDIO", DEFAULT_AUDIO))
    if not candidate.is_file():
        pytest.skip(f"no Arabic sample audio at {candidate}")
    return candidate


@pytest.fixture(autouse=True)
def _require_opt_in() -> None:
    if os.environ.get("S2T_RUN_SLOW") != "1":
        pytest.skip("set S2T_RUN_SLOW=1 to run the real-model integration test")


def test_real_model_transcribes_arabic(make_client, monkeypatch) -> None:
    audio = _audio_path()
    # "small" keeps the download to ~0.5 GB; override with S2T_TEST_MODEL.
    model = os.environ.get("S2T_TEST_MODEL", "small")

    monkeypatch.setenv("S2T_MODEL", model)
    get_settings.cache_clear()
    transcriber = Transcriber(get_settings())

    with make_client(transcriber, S2T_MODEL=model) as client:
        job_id = post_audio(client, audio, model=model).json()["job_id"]
        job = wait_for(client, job_id, "done", "error", timeout=45 * 60)

        assert job["status"] == "done", job.get("error_detail")
        text = job["text"]
        print(f"\n--- {model} on {audio.name} ---\n{text[:400]}\n")

        assert ARABIC_LETTERS.search(text), "no Arabic characters came back"
        assert job["segment_count"] > 0
        assert job["info"]["language"] == "ar"

        srt = client.get(f"/api/jobs/{job_id}/download?fmt=srt")
        assert srt.status_code == 200
        assert "-->" in srt.text

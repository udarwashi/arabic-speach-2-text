"""Route contracts, end to end against a fake transcriber but real ffmpeg."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.formats import RLM
from app.transcriber import Segment, TranscriptionError
from tests.conftest import (
    FAKE_SEGMENTS,
    FakeTranscriber,
    post_audio,
    wait_for,
    write_tone,
)

EXPECTED_TEXT = " ".join(segment.text for segment in FAKE_SEGMENTS)


# -- pages and metadata ----------------------------------------------------


def test_index_is_served_right_to_left(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'dir="rtl"' in response.text
    assert 'lang="ar"' in response.text


def test_static_assets_are_served(client) -> None:
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_models_endpoint_describes_the_dropdowns(client) -> None:
    payload = client.get("/api/models").json()
    keys = [model["key"] for model in payload["models"]]
    assert payload["default_model"] in keys
    assert {"large-v3-turbo", "large-v3", "small"} <= set(keys)
    assert ".mp3" in payload["extensions"]
    assert payload["max_upload_mb"] > 0
    assert {task["key"] for task in payload["tasks"]} == {"transcribe", "translate"}
    assert {lang["key"] for lang in payload["languages"]} == {"ar", "auto"}


def test_health_reports_runtime(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["runtime"]["device"] in {"cpu", "cuda"}
    assert payload["jobs"] == {"total": 0, "active": 0}


# -- happy path ------------------------------------------------------------


def test_transcribe_returns_a_job_then_completes(client, tone_wav: Path) -> None:
    response = post_audio(client, tone_wav)
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert accepted["filename"] == "tone.wav"

    job = wait_for(client, accepted["job_id"], "done")
    assert job["text"] == EXPECTED_TEXT
    assert job["segment_count"] == len(FAKE_SEGMENTS)
    assert job["progress"] == 1.0
    assert job["info"]["language"] == "ar"
    assert [s["index"] for s in job["segments"]] == [1, 2, 3]


def test_the_model_receives_a_16k_wav_not_the_upload(make_client, tone_wav: Path) -> None:
    fake = FakeTranscriber()
    with make_client(fake) as client:
        job_id = post_audio(client, tone_wav, model="large-v3").json()["job_id"]
        wait_for(client, job_id, "done")

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["path"].name.endswith(".16k.wav")
    assert call["model"] == "large-v3"
    assert call["language"] == "ar"
    assert call["task"] == "transcribe"
    assert fake.prepared == ["large-v3"]


def test_options_are_passed_through(make_client, tone_wav: Path) -> None:
    fake = FakeTranscriber()
    with make_client(fake) as client:
        job_id = post_audio(
            client, tone_wav, language="auto", task="translate", model="small"
        ).json()["job_id"]
        wait_for(client, job_id, "done")
    assert fake.calls[0]["language"] == "auto"
    assert fake.calls[0]["task"] == "translate"


def test_temporary_files_are_removed_after_a_job(make_client, tone_wav: Path, tmp_path) -> None:
    with make_client() as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        wait_for(client, job_id, "done")
    leftovers = list((tmp_path / "work").glob("*"))
    assert leftovers == []


def test_two_jobs_run_one_after_the_other(client, tone_wav: Path) -> None:
    first = post_audio(client, tone_wav).json()["job_id"]
    second = post_audio(client, tone_wav).json()["job_id"]
    assert wait_for(client, first, "done")["text"] == EXPECTED_TEXT
    assert wait_for(client, second, "done")["text"] == EXPECTED_TEXT


# -- streaming -------------------------------------------------------------


def test_stream_delivers_snapshot_segments_and_done(client, tone_wav: Path) -> None:
    job_id = post_audio(client, tone_wav).json()["job_id"]

    events: list[str] = []
    payloads: list[str] = []
    with client.stream("GET", f"/api/jobs/{job_id}/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event: "):
                events.append(line.removeprefix("event: "))
            elif line.startswith("data: "):
                payloads.append(line.removeprefix("data: "))
            if events and events[-1] in {"done", "error", "cancelled"}:
                break

    assert events[0] == "snapshot"
    assert events[-1] == "done"
    # Arabic must survive the wire unescaped.
    assert any("مرحبا" in payload for payload in payloads)


def test_stream_of_a_finished_job_replays_the_transcript(client, tone_wav: Path) -> None:
    job_id = post_audio(client, tone_wav).json()["job_id"]
    wait_for(client, job_id, "done")

    events: list[str] = []
    with client.stream("GET", f"/api/jobs/{job_id}/stream") as response:
        for line in response.iter_lines():
            if line.startswith("event: "):
                events.append(line.removeprefix("event: "))
            if events and events[-1] == "done":
                break
    assert events == ["snapshot", "done"]


def test_stream_of_an_unknown_job_is_404(client) -> None:
    assert client.get("/api/jobs/nope/stream").status_code == 404


# -- downloads -------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "marker"),
    [
        ("txt", "مرحبا بكم في هذا التسجيل"),
        ("srt", "00:00:00,000 --> 00:00:01,400"),
        ("vtt", "WEBVTT"),
    ],
)
def test_downloads(client, tone_wav: Path, fmt: str, marker: str) -> None:
    job_id = post_audio(client, tone_wav).json()["job_id"]
    wait_for(client, job_id, "done")

    response = client.get(f"/api/jobs/{job_id}/download?fmt={fmt}")
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert marker in body
    assert RLM in body  # right-to-left marks survive the download
    assert "attachment" in response.headers["content-disposition"]


def test_download_filename_survives_an_arabic_source_name(client, tmp_path: Path) -> None:
    arabic = write_tone(tmp_path / "تسجيل.wav")
    job_id = post_audio(client, arabic).json()["job_id"]
    wait_for(client, job_id, "done")

    disposition = client.get(f"/api/jobs/{job_id}/download?fmt=txt").headers[
        "content-disposition"
    ]
    # An ASCII fallback plus an RFC 5987 encoded name, so no header encoding error.
    assert 'filename="transcript-' in disposition
    assert "filename*=UTF-8''" in disposition
    assert disposition.isascii()


def test_download_rejects_an_unknown_format(client, tone_wav: Path) -> None:
    job_id = post_audio(client, tone_wav).json()["job_id"]
    wait_for(client, job_id, "done")
    assert client.get(f"/api/jobs/{job_id}/download?fmt=docx").status_code == 400


def test_download_before_there_is_any_text_is_409(make_client, tone_wav: Path) -> None:
    with make_client(FakeTranscriber(segments=[])) as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        wait_for(client, job_id, "done")
        assert client.get(f"/api/jobs/{job_id}/download?fmt=txt").status_code == 409


# -- validation and errors -------------------------------------------------


def test_unsupported_extension_is_415(client, tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("مرحبا", encoding="utf-8")
    assert post_audio(client, notes).status_code == 415


@pytest.mark.parametrize(
    "override", [{"model": "gpt-4"}, {"language": "fr"}, {"task": "summarise"}]
)
def test_bad_parameters_are_400(client, tone_wav: Path, override: dict) -> None:
    assert post_audio(client, tone_wav, **override).status_code == 400


def test_missing_file_is_422(client) -> None:
    assert client.post("/api/transcribe", data={"model": "small"}).status_code == 422


def test_empty_file_is_400(client, tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")
    assert post_audio(client, empty).status_code == 400


def test_oversized_upload_is_413_and_leaves_no_file(make_client, tmp_path: Path) -> None:
    big = tmp_path / "big.mp3"
    big.write_bytes(b"\x00" * (2 * 1024 * 1024))
    with make_client(S2T_MAX_UPLOAD_MB=1) as client:
        response = post_audio(client, big)
        assert response.status_code == 413
        assert client.get("/api/health").json()["jobs"]["total"] == 0
    assert list((tmp_path / "work").glob("*")) == []


def test_corrupt_audio_fails_the_job_with_a_message(client, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"definitely not an mp3" * 100)
    job_id = post_audio(client, corrupt).json()["job_id"]

    job = wait_for(client, job_id, "error")
    assert job["error"]
    assert job["error_detail"]
    assert job["segment_count"] == 0


def test_a_model_failure_fails_the_job(make_client, tone_wav: Path) -> None:
    broken = FakeTranscriber(error=TranscriptionError("فشل النموذج", "CUDA OOM"))
    with make_client(broken) as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        job = wait_for(client, job_id, "error")
    assert job["error"] == "فشل النموذج"
    assert job["error_detail"] == "CUDA OOM"


def test_a_model_download_failure_fails_the_job(make_client, tone_wav: Path) -> None:
    broken = FakeTranscriber(prepare_error=TranscriptionError("تعذّر التحميل"))
    with make_client(broken) as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        assert wait_for(client, job_id, "error")["error"] == "تعذّر التحميل"


def test_an_unexpected_error_still_reaches_a_terminal_state(
    make_client, tone_wav: Path
) -> None:
    broken = FakeTranscriber(error=RuntimeError("something odd"))
    with make_client(broken) as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        job = wait_for(client, job_id, "error")
    assert "something odd" in job["error_detail"]


def test_unknown_job_is_404(client) -> None:
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/jobs/missing/download").status_code == 404
    assert client.delete("/api/jobs/missing").status_code == 404


def test_delete_removes_the_job_and_reports_its_previous_state(
    client, tone_wav: Path
) -> None:
    job_id = post_audio(client, tone_wav).json()["job_id"]
    wait_for(client, job_id, "done")

    response = client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    # A finished job must not be reported as "cancelled".
    assert response.json() == {
        "job_id": job_id,
        "status": "deleted",
        "previous_status": "done",
    }
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_deleting_an_unfinished_job_cancels_it(make_client, tone_wav: Path) -> None:
    class Blocking(FakeTranscriber):
        def prepare(self, model: str) -> None:
            # Long enough that the job is still running when we delete it, short
            # enough that the worker thread does not stall shutdown.
            time.sleep(3)

    with make_client(Blocking()) as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        wait_for(client, job_id, "loading", "converting", "queued")

        response = client.delete(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["previous_status"] != "done"
        assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_segments_are_renumbered_from_one(make_client, tone_wav: Path) -> None:
    odd = FakeTranscriber(
        segments=[
            Segment(index=41, start=0.0, end=1.0, text="أ"),
            Segment(index=42, start=1.0, end=2.0, text="ب"),
        ]
    )
    with make_client(odd) as client:
        job_id = post_audio(client, tone_wav).json()["job_id"]
        job = wait_for(client, job_id, "done")
        srt = client.get(f"/api/jobs/{job_id}/download?fmt=srt").text

    assert job["text"] == "أ ب"
    assert [s["index"] for s in job["segments"]] == [41, 42]  # model indices are kept
    assert srt.startswith("1\n")  # but subtitle cues always start at 1

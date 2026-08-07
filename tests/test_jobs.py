"""Job registry lifecycle and the subscriber semantics the SSE endpoint relies on."""

from __future__ import annotations

import asyncio

import pytest

from app import jobs as jobs_module
from app.jobs import STATUS_LABELS, JobRegistry, JobStatus
from app.transcriber import Segment, TranscriptionInfo


def make_registry_with_job(**kwargs):
    registry = JobRegistry(**kwargs)
    job = registry.create(
        filename="clip.mp3", model="small", language="ar", task="transcribe"
    )
    return registry, job


def segment(index: int, start: float, end: float, text: str = "نص") -> Segment:
    return Segment(index=index, start=start, end=end, text=text)


def test_every_status_has_an_arabic_label() -> None:
    assert set(STATUS_LABELS) == set(JobStatus)
    assert all(label.strip() for label in STATUS_LABELS.values())


def test_create_get_and_drop() -> None:
    registry, job = make_registry_with_job()
    assert registry.get(job.id) is job
    assert registry.drop(job.id) is True
    assert registry.get(job.id) is None
    assert registry.drop(job.id) is False


def test_job_ids_are_unique() -> None:
    registry = JobRegistry()
    ids = {
        registry.create(filename="a", model="small", language="ar", task="transcribe").id
        for _ in range(50)
    }
    assert len(ids) == 50


def test_progress_is_zero_without_a_duration() -> None:
    _, job = make_registry_with_job()
    job.segments.append(segment(1, 0.0, 5.0))
    assert job.progress == 0.0


def test_progress_tracks_the_last_segment_end() -> None:
    registry, job = make_registry_with_job()
    registry.set_duration(job, 100.0)
    registry.add_segment(job, segment(1, 0.0, 25.0))
    assert job.progress == pytest.approx(0.25)


def test_progress_is_clamped_when_vad_overshoots_the_duration() -> None:
    registry, job = make_registry_with_job()
    registry.set_duration(job, 10.0)
    registry.add_segment(job, segment(1, 0.0, 12.0))
    assert job.progress == 1.0


def test_progress_is_one_when_done() -> None:
    registry, job = make_registry_with_job()
    registry.finish(job)
    assert job.progress == 1.0


def test_as_dict_can_omit_segments() -> None:
    registry, job = make_registry_with_job()
    registry.add_segment(job, segment(1, 0.0, 1.0))
    payload = job.as_dict(include_segments=False)
    assert "segments" not in payload and "text" not in payload
    assert payload["segment_count"] == 1
    assert payload["status_label"] == STATUS_LABELS[JobStatus.QUEUED]


def test_set_info_fills_in_the_duration() -> None:
    registry, job = make_registry_with_job()
    registry.set_info(
        job,
        TranscriptionInfo(
            language="ar",
            language_probability=0.99,
            duration=42.0,
            model="small",
            device="cpu",
            compute_type="int8",
        ),
    )
    assert job.duration == 42.0
    assert job.as_dict()["info"]["language"] == "ar"


def test_terminal_statuses() -> None:
    assert JobStatus.DONE.is_terminal
    assert JobStatus.ERROR.is_terminal
    assert JobStatus.CANCELLED.is_terminal
    assert not JobStatus.QUEUED.is_terminal
    assert not JobStatus.LOADING.is_terminal
    assert not JobStatus.TRANSCRIBING.is_terminal


async def test_subscriber_receives_snapshot_then_live_events() -> None:
    registry, job = make_registry_with_job()
    registry.set_duration(job, 10.0)

    stream = registry.subscribe(job.id)
    first = await anext(stream)
    assert first["event"] == "snapshot"
    assert first["data"]["segments"] == []

    registry.add_segment(job, segment(1, 0.0, 5.0, "مرحبا"))
    second = await anext(stream)
    assert second["event"] == "segment"
    assert second["data"]["segment"]["text"] == "مرحبا"
    assert second["data"]["progress"] == pytest.approx(0.5)

    registry.finish(job)
    third = await anext(stream)
    assert third["event"] == "done"
    assert third["data"]["text"] == "مرحبا"

    # The stream closes itself once a terminal event has been delivered.
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert job.subscribers == []


async def test_late_subscriber_sees_existing_segments_in_the_snapshot() -> None:
    registry, job = make_registry_with_job()
    registry.set_duration(job, 10.0)
    registry.add_segment(job, segment(1, 0.0, 2.0, "أول"))
    registry.add_segment(job, segment(2, 2.0, 4.0, "ثانٍ"))

    stream = registry.subscribe(job.id)
    snapshot = await anext(stream)
    assert [s["text"] for s in snapshot["data"]["segments"]] == ["أول", "ثانٍ"]

    registry.finish(job)
    assert (await anext(stream))["event"] == "done"
    await stream.aclose()


async def test_subscribing_to_a_finished_job_yields_snapshot_then_terminal() -> None:
    registry, job = make_registry_with_job()
    registry.add_segment(job, segment(1, 0.0, 1.0, "تم"))
    registry.finish(job)

    events = [event["event"] async for event in registry.subscribe(job.id)]
    assert events == ["snapshot", "done"]


async def test_subscribing_to_a_failed_job_reports_the_error() -> None:
    registry, job = make_registry_with_job()
    registry.fail(job, "فشل", "boom")

    events = [event async for event in registry.subscribe(job.id)]
    assert [event["event"] for event in events] == ["snapshot", "error"]
    assert events[-1]["data"]["error"] == "فشل"
    assert events[-1]["data"]["error_detail"] == "boom"


async def test_cancelled_job_terminates_the_stream() -> None:
    registry, job = make_registry_with_job()
    stream = registry.subscribe(job.id)
    await anext(stream)
    registry.cancel(job)
    event = await anext(stream)
    assert event["event"] == "cancelled"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_two_subscribers_both_receive_every_event() -> None:
    registry, job = make_registry_with_job()
    first = registry.subscribe(job.id)
    second = registry.subscribe(job.id)
    await anext(first)
    await anext(second)
    assert len(job.subscribers) == 2

    registry.add_segment(job, segment(1, 0.0, 1.0, "معاً"))
    assert (await anext(first))["event"] == "segment"
    assert (await anext(second))["event"] == "segment"

    await first.aclose()
    await second.aclose()
    assert job.subscribers == []


async def test_heartbeat_keeps_an_idle_stream_alive(monkeypatch) -> None:
    monkeypatch.setattr(jobs_module, "HEARTBEAT_SECONDS", 0.01)
    registry, job = make_registry_with_job()
    stream = registry.subscribe(job.id)
    await anext(stream)
    assert (await asyncio.wait_for(anext(stream), 2.0))["event"] == "heartbeat"
    await stream.aclose()


async def test_subscribing_to_an_unknown_job_raises() -> None:
    registry = JobRegistry()
    with pytest.raises(KeyError):
        await anext(registry.subscribe("does-not-exist"))


def test_purge_removes_only_expired_terminal_jobs() -> None:
    registry = JobRegistry(ttl_seconds=100)
    fresh_done = registry.create(
        filename="a", model="small", language="ar", task="transcribe"
    )
    stale_done = registry.create(
        filename="b", model="small", language="ar", task="transcribe"
    )
    stale_running = registry.create(
        filename="c", model="small", language="ar", task="transcribe"
    )

    registry.finish(fresh_done)
    registry.finish(stale_done)
    stale_done.updated_at -= 200
    stale_running.updated_at -= 200  # still running: must survive

    assert registry.purge_expired() == 1
    assert registry.get(stale_done.id) is None
    assert registry.get(fresh_done.id) is not None
    assert registry.get(stale_running.id) is not None

"""In-memory transcription jobs and the fan-out that feeds Server-Sent Events.

All mutation happens on the event loop, in the job worker coroutine. That is the
invariant the subscriber snapshot relies on: ``subscribe`` registers its queue and
takes its snapshot without awaiting in between, so it can neither miss an event
nor see one twice.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from .formats import full_text
from .transcriber import Segment, TranscriptionInfo


class JobStatus(str, Enum):
    QUEUED = "queued"
    CONVERTING = "converting"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)


# Arabic status labels, so the UI does not have to keep its own copy in sync.
STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.QUEUED: "في الانتظار…",
    JobStatus.CONVERTING: "جارٍ تحويل الملف الصوتي…",
    JobStatus.LOADING: "جارٍ تحضير النموذج (التنزيل يحدث مرة واحدة فقط)…",
    JobStatus.TRANSCRIBING: "جارٍ التعرّف على الكلام…",
    JobStatus.DONE: "تم التحويل بنجاح",
    JobStatus.ERROR: "حدث خطأ",
    JobStatus.CANCELLED: "تم الإلغاء",
}

HEARTBEAT_SECONDS = 15.0


@dataclass
class Job:
    """One transcription request and everything produced for it so far."""

    id: str
    filename: str
    model: str
    language: str
    task: str
    status: JobStatus = JobStatus.QUEUED
    segments: list[Segment] = field(default_factory=list)
    info: TranscriptionInfo | None = None
    duration: float | None = None
    error: str | None = None
    error_detail: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    task_handle: asyncio.Task | None = field(default=None, repr=False)
    subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

    @property
    def progress(self) -> float:
        """Fraction of the audio decoded so far, in ``[0, 1]``."""
        if self.status is JobStatus.DONE:
            return 1.0
        if not self.duration or not self.segments:
            return 0.0
        return min(1.0, max(0.0, self.segments[-1].end / self.duration))

    def as_dict(self, *, include_segments: bool = True) -> dict:
        payload = {
            "job_id": self.id,
            "filename": self.filename,
            "model": self.model,
            "language": self.language,
            "task": self.task,
            "status": self.status.value,
            "status_label": STATUS_LABELS[self.status],
            "duration": round(self.duration, 3) if self.duration else None,
            "progress": round(self.progress, 4),
            "segment_count": len(self.segments),
            "info": self.info.as_dict() if self.info else None,
            "error": self.error,
            "error_detail": self.error_detail,
        }
        if include_segments:
            payload["segments"] = [segment.as_dict() for segment in self.segments]
            payload["text"] = full_text(self.segments)
        return payload


class JobRegistry:
    """Holds jobs for the lifetime of the process (plus a TTL)."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._jobs: dict[str, Job] = {}
        self._ttl = ttl_seconds

    # -- lifecycle ---------------------------------------------------------

    def create(self, *, filename: str, model: str, language: str, task: str) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            filename=filename,
            model=model,
            language=language,
            task=task,
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def drop(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    # -- mutation, each publishing to subscribers --------------------------

    def set_status(self, job: Job, status: JobStatus) -> None:
        job.status = status
        job.updated_at = time.monotonic()
        self._publish(job, "status", job.as_dict(include_segments=False))

    def set_info(self, job: Job, info: TranscriptionInfo) -> None:
        job.info = info
        if info.duration:
            job.duration = info.duration
        job.updated_at = time.monotonic()
        self._publish(job, "status", job.as_dict(include_segments=False))

    def set_duration(self, job: Job, duration: float | None) -> None:
        job.duration = duration
        job.updated_at = time.monotonic()

    def add_segment(self, job: Job, segment: Segment) -> None:
        job.segments.append(segment)
        job.updated_at = time.monotonic()
        self._publish(
            job,
            "segment",
            {"segment": segment.as_dict(), "progress": round(job.progress, 4)},
        )

    def finish(self, job: Job) -> None:
        job.status = JobStatus.DONE
        job.updated_at = time.monotonic()
        self._publish(job, "done", job.as_dict())

    def fail(self, job: Job, message: str, detail: str = "") -> None:
        job.status = JobStatus.ERROR
        job.error = message
        job.error_detail = detail or None
        job.updated_at = time.monotonic()
        self._publish(job, "error", job.as_dict(include_segments=False))

    def cancel(self, job: Job) -> None:
        job.status = JobStatus.CANCELLED
        job.updated_at = time.monotonic()
        self._publish(job, "cancelled", job.as_dict(include_segments=False))

    def _publish(self, job: Job, event: str, data: dict) -> None:
        message = {"event": event, "data": data}
        for queue in list(job.subscribers):
            queue.put_nowait(message)

    # -- subscription ------------------------------------------------------

    async def subscribe(self, job_id: str):
        """Yield a snapshot, then live events, then stop at the terminal event.

        Heartbeats are emitted by this generator rather than by the caller: a
        timeout applied from outside would cancel the pending ``queue.get()``
        and tear the generator down.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)

        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)
        try:
            yield {"event": "snapshot", "data": job.as_dict()}
            if job.status.is_terminal:
                yield {"event": _terminal_event(job), "data": job.as_dict()}
                return

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": {}}
                    continue
                yield message
                if message["event"] in ("done", "error", "cancelled"):
                    return
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    # -- housekeeping ------------------------------------------------------

    def purge_expired(self, *, now: float | None = None) -> int:
        """Drop finished jobs older than the TTL. Returns how many were removed."""
        current = time.monotonic() if now is None else now
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status.is_terminal and (current - job.updated_at) > self._ttl
        ]
        for job_id in expired:
            del self._jobs[job_id]
        return len(expired)


def _terminal_event(job: Job) -> str:
    if job.status is JobStatus.ERROR:
        return "error"
    if job.status is JobStatus.CANCELLED:
        return "cancelled"
    return "done"

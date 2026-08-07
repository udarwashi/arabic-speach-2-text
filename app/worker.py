"""The job pipeline: normalise the audio, then decode it segment by segment.

``faster-whisper`` is blocking, so each step hops to a worker thread. The loop
itself stays on the event loop, which means every publish happens there too and
the registry needs no cross-thread signalling.

Only one job decodes at a time (``semaphore``): a 4 GB GPU cannot hold two large
models, and two concurrent decodes on one model are not safe anyway.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterator

from .audio import AudioError, convert_to_wav
from .jobs import Job, JobRegistry, JobStatus
from .transcriber import Segment, SupportsTranscription, TranscriptionError

logger = logging.getLogger(__name__)

_EXHAUSTED = object()


async def run_job(
    *,
    registry: JobRegistry,
    job: Job,
    transcriber: SupportsTranscription,
    source_path: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """Drive one job to a terminal state. Never raises except on cancellation."""
    wav_path = source_path.with_suffix(".16k.wav")
    try:
        async with semaphore:
            registry.set_status(job, JobStatus.CONVERTING)
            info = await asyncio.to_thread(convert_to_wav, source_path, wav_path)
            registry.set_duration(job, info.duration)

            registry.set_status(job, JobStatus.LOADING)
            await asyncio.to_thread(transcriber.prepare, job.model)

            registry.set_status(job, JobStatus.TRANSCRIBING)
            transcription_info, segments = await asyncio.to_thread(
                transcriber.transcribe,
                wav_path,
                model=job.model,
                language=job.language,
                task=job.task,
            )
            registry.set_info(job, transcription_info)

            await _drain(registry, job, segments)
            registry.finish(job)

    except asyncio.CancelledError:
        registry.cancel(job)
        raise
    except AudioError as exc:
        logger.warning("job %s audio error: %s (%s)", job.id, exc.message, exc.detail)
        registry.fail(job, exc.message, exc.detail)
    except TranscriptionError as exc:
        logger.warning("job %s model error: %s (%s)", job.id, exc.message, exc.detail)
        registry.fail(job, exc.message, exc.detail)
    except Exception as exc:  # noqa: BLE001 - a job must always reach a terminal state
        logger.exception("job %s failed unexpectedly", job.id)
        registry.fail(job, "حدث خطأ غير متوقع أثناء المعالجة.", f"{type(exc).__name__}: {exc}")
    finally:
        _remove_quietly(source_path)
        _remove_quietly(wav_path)


async def _drain(
    registry: JobRegistry, job: Job, segments: Iterator[Segment]
) -> None:
    """Pull the lazy segment iterator one step at a time, off the event loop."""
    while True:
        segment = await asyncio.to_thread(_next_or_sentinel, segments)
        if segment is _EXHAUSTED:
            return
        registry.add_segment(job, segment)  # type: ignore[arg-type]


def _next_or_sentinel(iterator: Iterator[Segment]) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return _EXHAUSTED


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem edge case
        logger.debug("could not delete %s: %s", path, exc)

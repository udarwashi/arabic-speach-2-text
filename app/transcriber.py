"""The Whisper wrapper.

Owns the loaded models and the decoding options. The options are not defaults —
they are tuned for Arabic, and the reasoning for each is in the comments below.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from .config import MODEL_REGISTRY, Settings

logger = logging.getLogger(__name__)

# Retried in order when a decode looks degenerate (repetition loops, low
# confidence). Whisper hits these more often on Arabic than on English.
TEMPERATURE_LADDER = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class TranscriptionError(Exception):
    """Raised when a model cannot be loaded or a decode fails."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class Segment:
    """One timestamped chunk of transcript."""

    index: int
    start: float
    end: float
    text: str

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


@dataclass(frozen=True)
class TranscriptionInfo:
    """What the model reported about the audio before decoding it."""

    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    compute_type: str

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "duration": round(self.duration, 3),
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
        }


class SupportsTranscription(Protocol):
    """The surface the API depends on, so tests can supply a fake."""

    def prepare(self, model: str) -> None: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str,
        task: str,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]: ...

    def runtime_info(self) -> dict: ...


def cuda_device_count() -> int:
    """Number of usable CUDA devices, or 0 if CTranslate2 cannot see any."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # pragma: no cover - depends on host drivers
        logger.debug("CUDA probe failed: %s", exc)
        return 0


@dataclass
class Transcriber:
    """Loads Whisper models on demand and keeps them cached in memory."""

    settings: Settings
    _models: dict[tuple[str, str, str], object] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _warnings: list[str] = field(default_factory=list, init=False)

    # -- device / precision resolution -------------------------------------

    def resolve_device(self) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        return "cuda" if cuda_device_count() > 0 else "cpu"

    def resolve_compute_type(self, model_key: str, device: str) -> str:
        if self.settings.compute_type != "auto":
            return self.settings.compute_type
        if device == "cuda":
            return MODEL_REGISTRY[model_key].cuda_compute_type
        # int8 is the only sane choice on CPU: float32 large-v3 is unusably slow.
        return "int8"

    # -- model cache -------------------------------------------------------

    def load(self, model_key: str) -> tuple[object, str, str]:
        """Return ``(model, device, compute_type)``, loading and caching as needed."""
        if model_key not in MODEL_REGISTRY:
            raise TranscriptionError(
                f"النموذج غير معروف: {model_key}",
                f"unknown model key {model_key!r}",
            )

        device = self.resolve_device()
        compute_type = self.resolve_compute_type(model_key, device)

        with self._lock:
            cached = self._models.get((model_key, device, compute_type))
            if cached is not None:
                return cached, device, compute_type

            try:
                if device == "cuda":
                    self._evict_cuda_models()
                model = self._build(model_key, device, compute_type)
            except Exception as exc:
                if device != "cuda":
                    raise TranscriptionError(
                        "تعذّر تحميل نموذج التعرّف على الكلام.",
                        f"{type(exc).__name__}: {exc}",
                    ) from exc
                # A 4 GB card, a missing cuDNN, or a busy GPU should degrade to
                # CPU rather than fail the request outright.
                warning = (
                    f"GPU load of {model_key} failed ({type(exc).__name__}: {exc}); "
                    "falling back to CPU int8"
                )
                logger.warning(warning)
                self._remember_warning(warning)
                device, compute_type = "cpu", "int8"
                cached = self._models.get((model_key, device, compute_type))
                if cached is not None:
                    return cached, device, compute_type
                try:
                    model = self._build(model_key, device, compute_type)
                except Exception as cpu_exc:
                    raise TranscriptionError(
                        "تعذّر تحميل نموذج التعرّف على الكلام على المعالج أيضاً.",
                        f"{type(cpu_exc).__name__}: {cpu_exc}",
                    ) from cpu_exc

            self._models[(model_key, device, compute_type)] = model
            return model, device, compute_type

    def _evict_cuda_models(self) -> None:
        """Keep at most one model in VRAM.

        Two large models will not fit on a 4 GB card, so switching models on the
        GPU unloads the previous one instead of stacking allocations. Caller must
        hold ``_lock``.
        """
        stale = [key for key in self._models if key[1] == "cuda"]
        for key in stale:
            logger.info("unloading %s from GPU to make room", key[0])
            del self._models[key]
        if stale:
            gc.collect()  # CTranslate2 frees VRAM when the model is collected

    def _build(self, model_key: str, device: str, compute_type: str) -> object:
        from faster_whisper import WhisperModel

        logger.info("loading %s on %s (%s)", model_key, device, compute_type)
        return WhisperModel(
            model_key,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.settings.cpu_threads,
            download_root=str(self.settings.model_dir) if self.settings.model_dir else None,
        )

    def _remember_warning(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)

    def prepare(self, model: str) -> None:
        """Load (and on first use download) a model without decoding anything.

        Called as its own pipeline step so the UI can say "preparing the model"
        instead of appearing to hang for the length of a 1.6 GB download.
        """
        self.load(model)

    # -- transcription -----------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str,
        task: str,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        """Start a transcription.

        Returns immediately with the detected-language info plus a lazy iterator;
        the actual decoding happens as the iterator is consumed, which is what
        lets the API stream segments to the browser as they are produced.
        """
        whisper_model, device, compute_type = self.load(model)

        try:
            raw_segments, raw_info = whisper_model.transcribe(  # type: ignore[attr-defined]
                str(audio_path),
                # "auto" means let Whisper detect; otherwise pin it, because
                # detection is unreliable on short clips and on recitation.
                language=None if language == "auto" else language,
                task=task,
                beam_size=self.settings.beam_size,
                temperature=TEMPERATURE_LADDER,
                # Do not feed the previous window back in: this is the single
                # most effective guard against Arabic repetition loops.
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                # Trim silence so the model is not asked to invent text for it.
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400},
            )
        except Exception as exc:
            raise TranscriptionError(
                "فشل تشغيل نموذج التعرّف على الكلام.",
                f"{type(exc).__name__}: {exc}",
            ) from exc

        info = TranscriptionInfo(
            language=getattr(raw_info, "language", language) or language,
            language_probability=float(getattr(raw_info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(raw_info, "duration", 0.0) or 0.0),
            model=model,
            device=device,
            compute_type=compute_type,
        )
        return info, _clean_segments(raw_segments)

    # -- introspection -----------------------------------------------------

    def runtime_info(self) -> dict:
        device = self.resolve_device()
        with self._lock:
            loaded = [
                {"model": key, "device": dev, "compute_type": ct}
                for key, dev, ct in self._models
            ]
            warnings = list(self._warnings)
        return {
            "device": device,
            "cuda_devices": cuda_device_count(),
            "loaded_models": loaded,
            "warnings": warnings,
            "ffmpeg": os.environ.get("S2T_FFMPEG", "ffmpeg"),
        }


def _clean_segments(raw_segments: Iterator[object]) -> Iterator[Segment]:
    """Renumber segments and drop the empty ones VAD tends to leave behind."""
    index = 0
    for raw in raw_segments:
        text = (getattr(raw, "text", "") or "").strip()
        if not text:
            continue
        index += 1
        yield Segment(
            index=index,
            start=float(getattr(raw, "start", 0.0) or 0.0),
            end=float(getattr(raw, "end", 0.0) or 0.0),
            text=text,
        )

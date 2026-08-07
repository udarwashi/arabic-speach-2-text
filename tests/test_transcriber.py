"""Device/precision resolution, the model cache, and the Arabic decoding options."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app import transcriber as transcriber_module
from app.config import Settings
from app.transcriber import Segment, Transcriber, TranscriptionError, _clean_segments


def make_settings(**overrides) -> Settings:
    base = dict(
        default_model="large-v3-turbo",
        device="auto",
        compute_type="auto",
        beam_size=5,
        cpu_threads=0,
        max_upload_bytes=1024,
        job_ttl_seconds=60,
        work_dir=Path("/tmp"),
        model_dir=None,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(transcriber_module, "cuda_device_count", lambda: 0)


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(transcriber_module, "cuda_device_count", lambda: 1)


class StubModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, segments=(), *, fail: Exception | None = None) -> None:
        self.segments = list(segments)
        self.fail = fail
        self.kwargs: dict = {}

    def transcribe(self, path, **kwargs):
        if self.fail is not None:
            raise self.fail
        self.kwargs = kwargs
        info = SimpleNamespace(language="ar", language_probability=0.97, duration=12.5)
        return iter(self.segments), info


def install_stub(monkeypatch, transcriber: Transcriber, model: StubModel) -> list[tuple]:
    """Replace model construction, recording every (key, device, compute) built."""
    built: list[tuple] = []

    def fake_build(model_key, device, compute_type):
        built.append((model_key, device, compute_type))
        return model

    monkeypatch.setattr(transcriber, "_build", fake_build)
    return built


# -- device and precision --------------------------------------------------


def test_explicit_device_is_respected(with_cuda) -> None:
    assert Transcriber(make_settings(device="cpu")).resolve_device() == "cpu"


def test_auto_device_picks_cuda_when_present(with_cuda) -> None:
    assert Transcriber(make_settings()).resolve_device() == "cuda"


def test_auto_device_falls_back_to_cpu(no_cuda) -> None:
    assert Transcriber(make_settings()).resolve_device() == "cpu"


def test_large_v3_is_quantized_on_gpu_so_it_fits_4gb() -> None:
    worker = Transcriber(make_settings())
    assert worker.resolve_compute_type("large-v3", "cuda") == "int8_float16"
    assert worker.resolve_compute_type("large-v3-turbo", "cuda") == "float16"


def test_cpu_always_uses_int8() -> None:
    worker = Transcriber(make_settings())
    assert worker.resolve_compute_type("large-v3", "cpu") == "int8"
    assert worker.resolve_compute_type("small", "cpu") == "int8"


def test_explicit_compute_type_overrides_everything() -> None:
    worker = Transcriber(make_settings(compute_type="float32"))
    assert worker.resolve_compute_type("large-v3", "cuda") == "float32"
    assert worker.resolve_compute_type("small", "cpu") == "float32"


# -- model cache -----------------------------------------------------------


def test_models_are_built_once_and_reused(monkeypatch, no_cuda) -> None:
    worker = Transcriber(make_settings())
    built = install_stub(monkeypatch, worker, StubModel())

    first, *_ = worker.load("small")
    second, *_ = worker.load("small")

    assert first is second
    assert built == [("small", "cpu", "int8")]


def test_unknown_model_is_rejected(no_cuda) -> None:
    with pytest.raises(TranscriptionError, match="غير معروف"):
        Transcriber(make_settings()).load("whisper-xxl")


def test_switching_gpu_models_unloads_the_previous_one(monkeypatch, with_cuda) -> None:
    """Two large models will not fit in 4 GB, so only one may stay in VRAM."""
    worker = Transcriber(make_settings())
    built = install_stub(monkeypatch, worker, StubModel())

    worker.load("large-v3-turbo")
    assert len(worker.runtime_info()["loaded_models"]) == 1

    worker.load("large-v3")
    loaded = worker.runtime_info()["loaded_models"]
    assert len(loaded) == 1
    assert loaded[0]["model"] == "large-v3"
    assert built == [
        ("large-v3-turbo", "cuda", "float16"),
        ("large-v3", "cuda", "int8_float16"),
    ]


def test_cpu_models_are_kept_side_by_side(monkeypatch, no_cuda) -> None:
    worker = Transcriber(make_settings())
    install_stub(monkeypatch, worker, StubModel())
    worker.load("small")
    worker.load("large-v3")
    assert len(worker.runtime_info()["loaded_models"]) == 2


def test_gpu_failure_falls_back_to_cpu_and_warns(monkeypatch, with_cuda) -> None:
    worker = Transcriber(make_settings())
    attempts: list[str] = []

    def flaky_build(model_key, device, compute_type):
        attempts.append(device)
        if device == "cuda":
            raise RuntimeError("cudnn missing")
        return StubModel()

    monkeypatch.setattr(worker, "_build", flaky_build)

    _, device, compute_type = worker.load("large-v3-turbo")
    assert (device, compute_type) == ("cpu", "int8")
    assert attempts == ["cuda", "cpu"]

    warnings = worker.runtime_info()["warnings"]
    assert len(warnings) == 1 and "cudnn missing" in warnings[0]


def test_cpu_failure_is_reported_as_a_transcription_error(monkeypatch, no_cuda) -> None:
    worker = Transcriber(make_settings())
    monkeypatch.setattr(
        worker, "_build", lambda *a: (_ for _ in ()).throw(RuntimeError("no disk"))
    )
    with pytest.raises(TranscriptionError) as excinfo:
        worker.load("small")
    assert "no disk" in excinfo.value.detail


# -- decoding options ------------------------------------------------------


def test_arabic_decoding_options_are_applied(monkeypatch, no_cuda) -> None:
    """These options are the accuracy contract; a regression here is silent."""
    stub = StubModel()
    worker = Transcriber(make_settings(beam_size=7))
    install_stub(monkeypatch, worker, stub)

    worker.transcribe(Path("a.wav"), model="small", language="ar", task="transcribe")

    assert stub.kwargs["language"] == "ar"
    assert stub.kwargs["beam_size"] == 7
    # The guard against Whisper's Arabic repetition loops.
    assert stub.kwargs["condition_on_previous_text"] is False
    assert stub.kwargs["vad_filter"] is True
    assert stub.kwargs["temperature"] == transcriber_module.TEMPERATURE_LADDER
    assert stub.kwargs["task"] == "transcribe"


def test_auto_language_is_sent_as_none(monkeypatch, no_cuda) -> None:
    stub = StubModel()
    worker = Transcriber(make_settings())
    install_stub(monkeypatch, worker, stub)

    info, _ = worker.transcribe(
        Path("a.wav"), model="small", language="auto", task="transcribe"
    )
    assert stub.kwargs["language"] is None
    assert info.language == "ar"  # whatever the model detected


def test_transcription_info_is_normalised(monkeypatch, no_cuda) -> None:
    worker = Transcriber(make_settings())
    install_stub(monkeypatch, worker, StubModel())

    info, _ = worker.transcribe(
        Path("a.wav"), model="small", language="ar", task="transcribe"
    )
    assert info.as_dict() == {
        "language": "ar",
        "language_probability": 0.97,
        "duration": 12.5,
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
    }


def test_a_decode_failure_becomes_a_transcription_error(monkeypatch, no_cuda) -> None:
    worker = Transcriber(make_settings())
    install_stub(monkeypatch, worker, StubModel(fail=RuntimeError("bad audio")))
    with pytest.raises(TranscriptionError) as excinfo:
        worker.transcribe(Path("a.wav"), model="small", language="ar", task="transcribe")
    assert "bad audio" in excinfo.value.detail


# -- segment cleaning ------------------------------------------------------


def test_clean_segments_renumbers_strips_and_drops_blanks() -> None:
    raw = [
        SimpleNamespace(start=0.0, end=1.0, text="  مرحبا  "),
        SimpleNamespace(start=1.0, end=2.0, text="   "),
        SimpleNamespace(start=2.0, end=3.0, text="بكم"),
    ]
    cleaned = list(_clean_segments(iter(raw)))
    assert cleaned == [
        Segment(index=1, start=0.0, end=1.0, text="مرحبا"),
        Segment(index=2, start=2.0, end=3.0, text="بكم"),
    ]


def test_clean_segments_tolerates_missing_attributes() -> None:
    cleaned = list(_clean_segments(iter([SimpleNamespace(text="نص")])))
    assert cleaned == [Segment(index=1, start=0.0, end=0.0, text="نص")]


def test_clean_segments_is_lazy() -> None:
    """Laziness is what lets the API stream text as it is decoded."""
    consumed: list[int] = []

    def source():
        for n in range(3):
            consumed.append(n)
            yield SimpleNamespace(start=float(n), end=n + 1.0, text=f"نص{n}")

    stream = _clean_segments(source())
    next(stream)
    assert consumed == [0]  # the rest has not been decoded yet

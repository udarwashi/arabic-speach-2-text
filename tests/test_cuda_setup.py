"""The on-demand CUDA runtime fetch.

These tests run on Linux, where none of the real machinery can execute, so the
Windows-only pieces (``ctypes.WinDLL``, ``os.add_dll_directory``) are faked. What
is exercised for real is the policy: when do we download, when do we refuse, and
what happens when a download is corrupt.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from app import cuda_setup


def _wheel_bytes() -> bytes:
    """A stand-in wheel with the same member layout as the NVIDIA ones."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nvidia/cudnn/bin/cudnn_ops64_9.dll", b"fake-dll-one")
        archive.writestr("nvidia/cudnn/bin/cudnn_graph64_9.dll", b"fake-dll-two")
        archive.writestr("nvidia/cudnn/include/cudnn.h", b"not a dll")
        archive.writestr("nvidia_cudnn_cu12.dist-info/METADATA", b"metadata")
    return buffer.getvalue()


def _stub_stream(payload: bytes, *, honour_range: bool = True):
    """Stand in for the network: serve ``payload`` from the requested offset."""

    def stream(url, handle, size, log, offset=0):
        if offset and not honour_range:
            return False
        handle.write(payload[offset:])
        return True

    return stream


@pytest.fixture
def fake_wheel(tmp_path: Path) -> Path:
    path = tmp_path / "fake.whl"
    path.write_bytes(_wheel_bytes())
    return path


def _pretend_windows(monkeypatch, *, driver: bool) -> None:
    monkeypatch.setattr(cuda_setup.sys, "platform", "win32")

    def win_dll(name: str):
        if driver:
            return object()
        raise OSError(f"cannot load {name}")

    monkeypatch.setattr(ctypes, "WinDLL", win_dll, raising=False)


# -- driver detection ------------------------------------------------------


def test_no_driver_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(cuda_setup.sys, "platform", "linux")
    assert cuda_setup.has_nvidia_driver() is False


def test_driver_detected_when_nvcuda_loads(monkeypatch) -> None:
    _pretend_windows(monkeypatch, driver=True)
    assert cuda_setup.has_nvidia_driver() is True


def test_no_driver_when_nvcuda_missing(monkeypatch) -> None:
    _pretend_windows(monkeypatch, driver=False)
    assert cuda_setup.has_nvidia_driver() is False


# -- ensure_cuda policy ----------------------------------------------------


def test_skips_when_env_flag_set(monkeypatch, tmp_path: Path) -> None:
    _pretend_windows(monkeypatch, driver=True)
    monkeypatch.setenv("S2T_SKIP_CUDA", "1")
    assert cuda_setup.ensure_cuda(tmp_path / "cuda") is False
    assert not (tmp_path / "cuda").exists()


def test_zero_does_not_count_as_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cuda_setup.sys, "platform", "linux")
    monkeypatch.setenv("S2T_SKIP_CUDA", "0")
    # Falls through to the driver check rather than short-circuiting.
    assert cuda_setup.ensure_cuda(tmp_path / "cuda") is False


def test_returns_false_without_a_driver(monkeypatch, tmp_path: Path) -> None:
    _pretend_windows(monkeypatch, driver=False)
    assert cuda_setup.ensure_cuda(tmp_path / "cuda") is False


def test_populated_cache_is_registered_without_downloading(
    monkeypatch, tmp_path: Path
) -> None:
    _pretend_windows(monkeypatch, driver=True)
    target = tmp_path / "cuda"
    target.mkdir()
    (target / "cudnn_ops64_9.dll").write_bytes(b"x")
    (target / cuda_setup.MARKER).write_text(cuda_setup.marker_text(), encoding="utf-8")

    registered: list[str] = []
    monkeypatch.setattr(cuda_setup, "_register", lambda path, log: registered.append(str(path)) or True)
    monkeypatch.setattr(
        cuda_setup, "_fetch", lambda *a, **k: pytest.fail("should not download")
    )

    assert cuda_setup.ensure_cuda(target) is True
    assert registered == [str(target)]


def test_stale_marker_triggers_a_refetch(monkeypatch, tmp_path: Path) -> None:
    _pretend_windows(monkeypatch, driver=True)
    target = tmp_path / "cuda"
    target.mkdir()
    (target / cuda_setup.MARKER).write_text("cudnn=0.0.0", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(
        cuda_setup, "_fetch", lambda wheel, dest, log: calls.append(wheel.name) or False
    )
    assert cuda_setup.ensure_cuda(target) is False
    assert calls  # it tried


def test_a_failed_download_leaves_no_marker(monkeypatch, tmp_path: Path) -> None:
    _pretend_windows(monkeypatch, driver=True)
    target = tmp_path / "cuda"
    monkeypatch.setattr(cuda_setup, "_fetch", lambda wheel, dest, log: False)

    assert cuda_setup.ensure_cuda(target) is False
    assert not (target / cuda_setup.MARKER).exists()


def test_ensure_cuda_never_raises(monkeypatch, tmp_path: Path) -> None:
    _pretend_windows(monkeypatch, driver=True)

    def explode(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(cuda_setup, "_fetch", explode)
    assert cuda_setup.ensure_cuda(tmp_path / "cuda") is False


# -- extraction ------------------------------------------------------------


def test_extract_takes_only_dlls_and_flattens_them(
    fake_wheel: Path, tmp_path: Path
) -> None:
    target = tmp_path / "out"
    count = cuda_setup._extract_dlls(fake_wheel, target)

    assert count == 2
    assert sorted(p.name for p in target.iterdir()) == [
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
    ]
    assert (target / "cudnn_ops64_9.dll").read_bytes() == b"fake-dll-one"


def test_extract_rejects_a_wheel_with_no_dlls(tmp_path: Path) -> None:
    empty = tmp_path / "empty.whl"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("nvidia/cudnn/include/cudnn.h", b"header only")

    assert cuda_setup._extract_dlls(empty, tmp_path / "out") == 0


# -- hashing ---------------------------------------------------------------


def test_download_rejects_a_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    payload = _wheel_bytes()
    wheel = cuda_setup.Wheel(
        name="fake.whl",
        url="https://example.invalid/fake.whl",
        sha256="0" * 64,  # deliberately wrong
        size=len(payload),
    )
    monkeypatch.setattr(cuda_setup, "_stream", _stub_stream(payload))

    assert cuda_setup._fetch(wheel, tmp_path, None) is False
    assert list(tmp_path.glob("*")) == []  # the .part file is cleaned up


def test_download_accepts_a_matching_hash(monkeypatch, tmp_path: Path) -> None:
    payload = _wheel_bytes()
    wheel = cuda_setup.Wheel(
        name="fake.whl",
        url="https://example.invalid/fake.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    monkeypatch.setattr(cuda_setup, "_stream", _stub_stream(payload))

    assert cuda_setup._fetch(wheel, tmp_path, None) is True
    assert (tmp_path / "cudnn_ops64_9.dll").exists()
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.whl"))


def test_download_resumes_a_partial_file(monkeypatch, tmp_path: Path) -> None:
    payload = _wheel_bytes()
    wheel = cuda_setup.Wheel(
        name="fake.whl",
        url="https://example.invalid/fake.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    part = tmp_path / "fake.whl.part"
    part.write_bytes(payload[:100])  # a previous attempt, interrupted

    seen: list[int] = []

    def stream(url, handle, size, log, offset=0):
        seen.append(offset)
        handle.write(payload[offset:])
        return True

    monkeypatch.setattr(cuda_setup, "_stream", stream)

    assert cuda_setup._download(wheel, part, None) == wheel.sha256
    assert seen == [100]  # it asked for the remainder, not the whole file


def test_download_restarts_when_the_server_ignores_the_range(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _wheel_bytes()
    wheel = cuda_setup.Wheel(
        name="fake.whl",
        url="https://example.invalid/fake.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    part = tmp_path / "fake.whl.part"
    part.write_bytes(payload[:100])

    monkeypatch.setattr(cuda_setup, "_stream", _stub_stream(payload, honour_range=False))

    # Falls back to a clean download rather than appending to the partial file.
    assert cuda_setup._download(wheel, part, None) == wheel.sha256


def test_download_restarts_on_a_416(monkeypatch, tmp_path: Path) -> None:
    from urllib.error import HTTPError

    payload = _wheel_bytes()
    wheel = cuda_setup.Wheel(
        name="fake.whl",
        url="https://example.invalid/fake.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    part = tmp_path / "fake.whl.part"
    part.write_bytes(payload + b"trailing junk")  # longer than the real file

    def stream(url, handle, size, log, offset=0):
        if offset:
            raise HTTPError(url, 416, "Range Not Satisfiable", {}, None)
        handle.write(payload)
        return True

    monkeypatch.setattr(cuda_setup, "_stream", stream)

    assert cuda_setup._download(wheel, part, None) == wheel.sha256


def test_a_stale_partial_from_another_build_is_caught_by_the_hash(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _wheel_bytes()
    wheel = cuda_setup.Wheel(
        name="fake.whl",
        url="https://example.invalid/fake.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    part = tmp_path / "fake.whl.part"
    part.write_bytes(b"bytes from a completely different file")

    monkeypatch.setattr(cuda_setup, "_stream", _stub_stream(payload))

    assert cuda_setup._fetch(wheel, tmp_path, None) is False
    assert not part.exists()


# -- the pinned table ------------------------------------------------------


def test_pinned_wheels_match_requirements_gpu() -> None:
    """The GPU path must be the version combination already tested on the 3050."""
    versions = {wheel.name.split("-")[1] for wheel in cuda_setup.WHEELS}
    assert versions == {"9.1.1.17", "12.4.5.8"}
    for wheel in cuda_setup.WHEELS:
        assert len(wheel.sha256) == 64
        assert wheel.url.startswith("https://files.pythonhosted.org/")
        assert wheel.name.endswith("win_amd64.whl")


# -- is the runtime actually usable? ---------------------------------------


def test_runtime_always_available_off_windows(monkeypatch) -> None:
    """Elsewhere the CUDA libraries arrive with CTranslate2, via pip."""
    monkeypatch.setattr(cuda_setup.sys, "platform", "linux")
    assert cuda_setup.cuda_runtime_available() is True


def test_runtime_available_when_every_dll_loads(monkeypatch) -> None:
    monkeypatch.setattr(cuda_setup.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: object(), raising=False)
    assert cuda_setup.cuda_runtime_available() is True


def test_runtime_unavailable_when_cublas_is_missing(monkeypatch) -> None:
    """The exact failure seen in the packaged build before this check existed."""
    monkeypatch.setattr(cuda_setup.sys, "platform", "win32")

    def win_dll(name: str):
        if name == "cublas64_12.dll":
            raise OSError("not found")
        return object()

    monkeypatch.setattr(ctypes, "WinDLL", win_dll, raising=False)
    assert cuda_setup.cuda_runtime_available() is False


def test_register_also_puts_the_directory_on_path(monkeypatch, tmp_path: Path) -> None:
    """CTranslate2 opens the CUDA libraries by name, and that search reads PATH."""
    added: list[str] = []
    monkeypatch.setattr(os := cuda_setup.os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")

    assert cuda_setup._register(tmp_path, None) is True
    assert added == [str(tmp_path)]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path)


def test_register_does_not_duplicate_the_path_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cuda_setup.os, "add_dll_directory", lambda p: None, raising=False)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin")

    cuda_setup._register(tmp_path, None)
    assert cuda_setup.os.environ["PATH"].count(str(tmp_path)) == 1

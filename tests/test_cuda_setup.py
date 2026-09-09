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
    monkeypatch.setattr(cuda_setup, "_stream", lambda url, handle, size, log: handle.write(payload))

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
    monkeypatch.setattr(cuda_setup, "_stream", lambda url, handle, size, log: handle.write(payload))

    assert cuda_setup._fetch(wheel, tmp_path, None) is True
    assert (tmp_path / "cudnn_ops64_9.dll").exists()
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.whl"))


# -- the pinned table ------------------------------------------------------


def test_pinned_wheels_match_requirements_gpu() -> None:
    """The GPU path must be the version combination already tested on the 3050."""
    versions = {wheel.name.split("-")[1] for wheel in cuda_setup.WHEELS}
    assert versions == {"9.1.1.17", "12.4.5.8"}
    for wheel in cuda_setup.WHEELS:
        assert len(wheel.sha256) == 64
        assert wheel.url.startswith("https://files.pythonhosted.org/")
        assert wheel.name.endswith("win_amd64.whl")

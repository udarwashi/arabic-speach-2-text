"""The frozen entry point.

The launcher's whole job is to arrange the process correctly *before* anything
else in the app is imported, so most of these tests are about ordering and paths
rather than behaviour.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from app import launcher


# -- locating the bundle ---------------------------------------------------


def test_bundle_dir_is_the_repo_when_not_frozen() -> None:
    assert (launcher.bundle_dir() / "app" / "launcher.py").is_file()


def test_bundle_dir_follows_meipass_when_frozen(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert launcher.bundle_dir() == tmp_path


# -- locating writable storage ---------------------------------------------


def test_data_dir_uses_localappdata_when_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert launcher.data_dir() == tmp_path / launcher.APP_DIR_NAME


def test_data_dir_falls_back_off_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert launcher.data_dir() == tmp_path / ".local" / "share" / launcher.APP_DIR_NAME


def test_data_dir_is_never_inside_the_bundle(monkeypatch, tmp_path: Path) -> None:
    """The bundle is temporary; a 1.6 GB model must not be written into it."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert launcher.bundle_dir() not in launcher.data_dir().parents


# -- environment preparation ------------------------------------------------


@pytest.fixture
def prepared(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    for name in ("S2T_MODEL_DIR", "S2T_WORK_DIR", "S2T_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    return launcher.configure_environment()


def test_configure_creates_every_directory(prepared: Path) -> None:
    for name in ("models", "work", "logs", "cuda"):
        assert (prepared / name).is_dir()


def test_configure_points_the_app_at_writable_storage(prepared: Path, monkeypatch) -> None:
    import os

    assert os.environ["S2T_MODEL_DIR"] == str(prepared / "models")
    assert os.environ["S2T_WORK_DIR"] == str(prepared / "work")


def test_configure_respects_an_explicit_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("S2T_MODEL_DIR", "/somewhere/else")
    launcher.configure_environment()
    import os

    assert os.environ["S2T_MODEL_DIR"] == "/somewhere/else"


def test_configure_puts_bundled_ffmpeg_first_on_path(
    monkeypatch, tmp_path: Path
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    monkeypatch.setattr(launcher, "bundle_dir", lambda: bundle)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin")

    launcher.configure_environment()

    import os

    assert os.environ["PATH"].split(os.pathsep)[0] == str(bundle / "bin")


def test_configure_leaves_path_alone_without_a_bundled_bin(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(launcher, "bundle_dir", lambda: tmp_path / "empty")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin")

    launcher.configure_environment()

    import os

    assert os.environ["PATH"] == "/usr/bin"


def test_settings_pick_up_the_prepared_directories(prepared: Path) -> None:
    """The ordering that matters: env first, then the lru_cached settings."""
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.model_dir == prepared / "models"
        assert settings.work_dir == prepared / "work"
    finally:
        get_settings.cache_clear()


def test_configure_disables_the_password_gate(prepared: Path) -> None:
    """A packaged single-user app must not present a login page."""
    import os

    assert os.environ["S2T_PASSWORD"] == ""


def test_configure_still_honours_an_explicit_password(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("S2T_PASSWORD", "hunter2")
    launcher.configure_environment()
    import os

    assert os.environ["S2T_PASSWORD"] == "hunter2"


# -- networking -------------------------------------------------------------


def test_free_port_returns_something_bindable() -> None:
    port = launcher.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_free_port_varies() -> None:
    assert len({launcher.free_port() for _ in range(5)}) > 1


def test_wait_until_ready_detects_a_listening_socket() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert launcher.wait_until_ready(port, timeout=2.0) is True


def test_wait_until_ready_gives_up_on_a_dead_port() -> None:
    port = launcher.free_port()
    assert launcher.wait_until_ready(port, timeout=0.5) is False


"""Desktop entry point: the program the packaged Windows executable runs.

The app itself is unchanged — a FastAPI server rendering its own frontend. What
this module adds is everything that stops being true once the code is frozen into
a bundle and handed to someone who has never opened a terminal:

* the bundle directory is temporary, so model weights and scratch audio are
  redirected to ``%LOCALAPPDATA%\\speech2text`` instead;
* ``ffmpeg`` is not installed on the user's machine, so the bundled copies are put
  on ``PATH`` where :mod:`app.audio` already looks for them;
* the CUDA runtime is fetched on demand by :mod:`app.cuda_setup`;
* the port is chosen by the OS, so "port 8000 is taken" cannot happen;
* the browser is opened for the user, and the console window is their quit button.

Ordering matters. ``configure_environment`` must run before anything imports
:mod:`app.config`, because ``get_settings`` is ``lru_cache``d and the first call
fixes the configuration for the life of the process. Hence the deferred imports.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Safe to import this early: it touches only the standard library and reads the
# environment when called, not when imported.
from app.cuda_setup import cuda_runtime_available, ensure_cuda, say

APP_DIR_NAME = "speech2text"
HOST = "127.0.0.1"
_READY_TIMEOUT = 60.0

_BANNER = """
==============================================================
  تحويل الصوت إلى نص  —  Arabic speech to text

  {url}

  التطبيق يعمل الآن، وقد فُتحت الصفحة في المتصفح.
  أبقِ هذه النافذة مفتوحة أثناء الاستخدام،
  وأغلقها عند الانتهاء لإيقاف البرنامج.
==============================================================
"""


def bundle_dir() -> Path:
    """Where this build's own files live.

    Under PyInstaller that is the unpacked bundle (``sys._MEIPASS``); running from
    a checkout it is the repository root.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Where the user's data lives — always writable, never inside the bundle."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIR_NAME
    return Path.home() / ".local" / "share" / APP_DIR_NAME


def configure_environment() -> Path:
    """Prepare directories, settings and ``PATH``. Returns the data directory."""
    root = data_dir()
    for name in ("models", "work", "logs", "cuda"):
        (root / name).mkdir(parents=True, exist_ok=True)

    # setdefault, not assignment: an explicit S2T_* in the environment still wins,
    # which keeps the packaged app debuggable.
    os.environ.setdefault("S2T_MODEL_DIR", str(root / "models"))
    os.environ.setdefault("S2T_WORK_DIR", str(root / "work"))

    # The desktop build is one person's app on their own loopback interface, where
    # a shared password protects nothing. Saying so explicitly beats relying on no
    # .env happening to be inside the bundle: config.load_dotenv only fills in keys
    # that are absent, so an empty value here settles it either way.
    os.environ.setdefault("S2T_PASSWORD", "")

    # app.audio resolves ffmpeg and ffprobe through PATH. That is the right
    # mechanism here precisely because they are launched as subprocesses — unlike
    # the CUDA DLLs, which need os.add_dll_directory instead.
    bundled_bin = bundle_dir() / "bin"
    if bundled_bin.is_dir():
        os.environ["PATH"] = f"{bundled_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    return root


def configure_logging(log_dir: Path) -> logging.Logger:
    """Log to a rotating file, so a failure is diagnosable with no terminal."""
    handler = RotatingFileHandler(
        log_dir / "speech2text.log",
        maxBytes=2 << 20,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler(sys.stderr))
    return logging.getLogger("speech2text.launcher")


def free_port() -> int:
    """Let the OS pick a port, so a busy one is never a failure mode."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def wait_until_ready(port: int, timeout: float = _READY_TIMEOUT) -> bool:
    """Block until the server accepts connections.

    Opening the browser first would show the user a connection-refused page.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _use_utf8_console() -> None:
    """The banner is Arabic; the Windows console default encoding is not."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _serve(port: int) -> threading.Thread:
    """Run uvicorn on a daemon thread. Imported late, after the environment is set."""
    import uvicorn

    from app.main import app

    config = uvicorn.Config(app, host=HOST, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    return thread


def main() -> int:
    _use_utf8_console()
    root = configure_environment()
    log = configure_logging(root / "logs")

    try:
        log.info("starting from %s, data in %s", bundle_dir(), root)

        on_gpu = ensure_cuda(root / "cuda", logger=log)
        log.info("CUDA runtime %s", "ready" if on_gpu else "not provisioned")

        # CTranslate2 reports a CUDA device whenever the driver can see the card,
        # whether or not cuBLAS is present, so resolve_device() would pick "cuda"
        # and then fail -- during decoding, where transcriber.py's fallback does
        # not reach, leaving the user with an error instead of a slower answer.
        # Decide it here, from whether the libraries can actually be loaded.
        if not cuda_runtime_available():
            os.environ.setdefault("S2T_DEVICE", "cpu")
            log.info("CUDA libraries are not loadable; using the CPU")

        port = free_port()
        url = f"http://{HOST}:{port}/"
        _serve(port)

        if wait_until_ready(port):
            log.info("serving on %s", url)
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - a missing browser is not fatal
                log.warning("could not open a browser; the URL is %s", url)
        else:
            log.error("the server did not start within %.0f seconds", _READY_TIMEOUT)
            say(f"تعذّر تشغيل الخادم. راجع ملف السجل: {root / 'logs'}")
            return _hold(1)

        say(_BANNER.format(url=url))
        threading.Event().wait()
        return 0
    except KeyboardInterrupt:
        log.info("stopped by the user")
        return 0
    except Exception:  # noqa: BLE001 - show the user something, never vanish
        log.exception("the application failed to start")
        say(f"\nحدث خطأ أثناء تشغيل البرنامج. التفاصيل في: {root / 'logs'}")
        return _hold(1)


def _hold(code: int) -> int:
    """Keep the console open so an error is readable after a double-click.

    The prompt goes through ``say`` first: ``input`` writes its prompt to stdout,
    and an encoding error there would replace the message the user needs with a
    traceback, in the one situation where something has already gone wrong.
    """
    say("\nاضغط Enter للإغلاق / press Enter to close ")
    try:
        input()
    except (EOFError, KeyboardInterrupt, OSError):
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())

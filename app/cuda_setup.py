"""Finding — and if necessary fetching — the CUDA runtime the GPU path needs.

CTranslate2 reaches an NVIDIA card through cuDNN and cuBLAS. Those two libraries
are about 1.5 GB installed, which is far too much to bundle into an installer for
a user who may well have no NVIDIA card at all. So they are downloaded on first
run, from PyPI, and only on a machine whose driver proves a card is present.

Nothing in this module is allowed to raise. A failure here simply means the app
runs on the CPU, which :mod:`app.transcriber` already falls back to on its own —
``resolve_device`` returns ``"cpu"`` when CTranslate2 reports no CUDA device.

The versions are pinned to the ones in ``requirements-gpu.txt``, so the GPU path
is the combination already measured on an RTX 3050.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.request import urlopen

__all__ = ["ensure_cuda", "has_nvidia_driver", "WHEELS", "MARKER"]

_LOG = logging.getLogger(__name__)

# Written once every wheel has been extracted, so a run interrupted midway is not
# mistaken for a usable cache.
MARKER = ".complete"

_CHUNK = 1 << 20  # 1 MiB
_DLL_PARENTS = ("bin",)  # inside a wheel the Windows DLLs live in nvidia/<lib>/bin/


@dataclass(frozen=True)
class Wheel:
    """One pinned NVIDIA wheel, verified by hash before it is ever opened."""

    name: str
    url: str
    sha256: str
    size: int

    @property
    def version(self) -> str:
        return self.name.split("-")[1]


WHEELS: tuple[Wheel, ...] = (
    Wheel(
        name="nvidia_cudnn_cu12-9.1.1.17-py3-none-win_amd64.whl",
        url=(
            "https://files.pythonhosted.org/packages/a5/0a/"
            "4a3852f359aa6043369217155c4b905226634025d1b9c605286fdb823847/"
            "nvidia_cudnn_cu12-9.1.1.17-py3-none-win_amd64.whl"
        ),
        sha256="d7c4b96b1c5ca8e4dc0bdbf2ce386903ba03faa94d2997b72769fa5048cb5f1f",
        size=679_907_000,
    ),
    Wheel(
        name="nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl",
        url=(
            "https://files.pythonhosted.org/packages/e2/2a/"
            "4f27ca96232e8b5269074a72e03b4e0d43aa68c9b965058b1684d07c6ff8/"
            "nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl"
        ),
        sha256="5a796786da89203a0657eda402bcdcec6180254a8ac22d72213abc42069522dc",
        size=396_900_000,
    ),
)


def marker_text() -> str:
    """The cache is only valid for the exact versions this build pins."""
    return " ".join(f"{wheel.name.split('-')[0]}={wheel.version}" for wheel in WHEELS)


def has_nvidia_driver() -> bool:
    """True when the NVIDIA display driver is installed.

    ``nvcuda.dll`` ships with that driver and with nothing else, which makes it a
    reliable, dependency-free signal that there is a card worth downloading a
    gigabyte for.
    """
    if sys.platform != "win32":
        return False
    try:
        ctypes.WinDLL("nvcuda.dll")
    except (OSError, AttributeError):
        return False
    return True


def ensure_cuda(target: Path, *, logger: logging.Logger | None = None) -> bool:
    """Make the CUDA DLLs importable, downloading them once if needed.

    Returns True only when ``target`` holds a complete set and has been added to
    the DLL search path. Every failure is logged and reported as False.
    """
    log = logger or _LOG
    try:
        return _ensure_cuda(target, log)
    except Exception:  # noqa: BLE001 - the CPU path must survive anything
        log.exception("preparing the CUDA runtime failed; continuing on the CPU")
        return False


def _ensure_cuda(target: Path, log: logging.Logger) -> bool:
    if os.environ.get("S2T_SKIP_CUDA", "").strip() not in ("", "0"):
        log.info("S2T_SKIP_CUDA is set — staying on the CPU")
        return False

    if not has_nvidia_driver():
        log.info("no NVIDIA driver found — running on the CPU")
        return False

    if _is_complete(target):
        log.info("CUDA runtime already present in %s", target)
        return _register(target, log)

    log.info(
        "NVIDIA driver found. Downloading the CUDA runtime once (~1.1 GB) into %s",
        target,
    )
    print(
        "\nتم العثور على كرت رسوميات NVIDIA — يجري تنزيل مكتبات التسريع مرة واحدة"
        " (حوالي 1.1 غيغابايت). لن يتكرر هذا في المرات القادمة.\n",
        flush=True,
    )

    target.mkdir(parents=True, exist_ok=True)
    for wheel in WHEELS:
        if not _fetch(wheel, target, log):
            log.warning("could not obtain %s — falling back to the CPU", wheel.name)
            return False

    (target / MARKER).write_text(marker_text(), encoding="utf-8")
    log.info("CUDA runtime ready")
    return _register(target, log)


def _is_complete(target: Path) -> bool:
    marker = target / MARKER
    try:
        return marker.read_text(encoding="utf-8").strip() == marker_text()
    except OSError:
        return False


def _register(target: Path, log: logging.Logger | None) -> bool:
    """Put ``target`` on the DLL search path.

    Since Python 3.8 extension modules no longer search ``PATH`` for their
    dependent DLLs, so this — not a ``PATH`` edit — is what makes cuDNN loadable.
    """
    add = getattr(os, "add_dll_directory", None)
    if add is None:  # not Windows; nothing to do and nothing that could use it
        return False
    try:
        add(str(target))
    except OSError:
        if log:
            log.warning("could not register %s on the DLL search path", target)
        return False
    return True


def _fetch(wheel: Wheel, target: Path, log: logging.Logger | None) -> bool:
    """Download one wheel, verify it, extract its DLLs, and clean up."""
    target.mkdir(parents=True, exist_ok=True)
    part = target / f"{wheel.name}.part"
    digest = hashlib.sha256()

    try:
        with part.open("wb") as handle:
            _stream(wheel.url, _HashingWriter(handle, digest), wheel.size, log)
    except OSError as exc:
        if log:
            log.warning("downloading %s failed: %s", wheel.name, exc)
        part.unlink(missing_ok=True)
        return False

    if digest.hexdigest() != wheel.sha256:
        if log:
            log.error(
                "%s failed its checksum (expected %s, got %s)",
                wheel.name,
                wheel.sha256,
                digest.hexdigest(),
            )
        part.unlink(missing_ok=True)
        return False

    try:
        extracted = _extract_dlls(part, target)
    finally:
        part.unlink(missing_ok=True)

    if extracted == 0:
        if log:
            log.error("%s contained no DLLs", wheel.name)
        return False
    if log:
        log.info("extracted %d DLLs from %s", extracted, wheel.name)
    return True


class _HashingWriter:
    """Hashes while writing, so the payload is never held in memory twice."""

    def __init__(self, handle: IO[bytes], digest) -> None:
        self._handle = handle
        self._digest = digest

    def write(self, chunk: bytes) -> int:
        self._digest.update(chunk)
        return self._handle.write(chunk)


def _stream(url: str, handle, size: int, log: logging.Logger | None) -> None:
    """Copy ``url`` into ``handle``, printing progress to the console."""
    with urlopen(url) as response:  # noqa: S310 - a pinned https URL
        total = int(response.headers.get("Content-Length") or size or 0)
        done = 0
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB"
                    f"  ({done * 100 // total}%)",
                    end="",
                    flush=True,
                )
    print(flush=True)


def _extract_dlls(archive_path: Path, target: Path) -> int:
    """Pull just the DLLs out of a wheel, flattened into ``target``.

    The headers, static libraries and metadata in these wheels are of no use to a
    running app and account for a fair share of their size.
    """
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = member.filename.split("/")
            if len(parts) < 2 or not parts[-1].lower().endswith(".dll"):
                continue
            if parts[-2].lower() not in _DLL_PARENTS:
                continue
            destination = target / Path(parts[-1]).name
            with archive.open(member) as source, destination.open("wb") as sink:
                while chunk := source.read(_CHUNK):
                    sink.write(chunk)
            count += 1
    return count

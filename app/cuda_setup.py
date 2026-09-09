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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

__all__ = [
    "ensure_cuda",
    "say",
    "has_nvidia_driver",
    "cuda_runtime_available",
    "WHEELS",
    "MARKER",
]

_LOG = logging.getLogger(__name__)

# Written once every wheel has been extracted, so a run interrupted midway is not
# mistaken for a usable cache.
MARKER = ".complete"

_CHUNK = 1 << 20  # 1 MiB
_DLL_PARENTS = ("bin",)  # inside a wheel the Windows DLLs live in nvidia/<lib>/bin/

# What CTranslate2 opens by name before it will run anything on the GPU.
_REQUIRED_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")


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


def say(message: str) -> None:
    """Print a status line without letting the console's encoding break anything.

    The Windows console defaults to a legacy code page that cannot represent
    Arabic, and an unguarded ``print`` there raises ``UnicodeEncodeError``. That
    once aborted the whole CUDA download over a cosmetic message, quietly costing
    the user their GPU, so every console write in this module goes through here.
    """
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            print(message.encode(encoding, "replace").decode(encoding, "replace"),
                  flush=True)
        except (UnicodeError, OSError):
            pass
    except OSError:
        pass


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
    say(
        "\nتم العثور على كرت رسوميات NVIDIA — يجري تنزيل مكتبات التسريع مرة واحدة"
        " (حوالي 1.1 غيغابايت). لن يتكرر هذا في المرات القادمة.\n"
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
    """Make the DLLs in ``target`` loadable, by both mechanisms that matter.

    ``os.add_dll_directory`` covers dependent-DLL resolution, which since Python
    3.8 no longer consults ``PATH``. But CTranslate2 opens the CUDA libraries by
    name at run time rather than importing them, and a plain ``LoadLibrary`` does
    still search ``PATH`` while ignoring the directories added that way. Doing
    both costs nothing and covers either code path.
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

    if str(target) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{target}{os.pathsep}{os.environ.get('PATH', '')}"
    return True


def cuda_runtime_available() -> bool:
    """True when the libraries CTranslate2 needs for CUDA can actually load.

    Checked after provisioning, because the answer decides whether the GPU is
    usable at all. ``transcriber`` only falls back to the CPU when *loading* a
    model raises; this failure surfaces later, during decoding, where nothing
    catches it — so the choice has to be made up front.

    Off Windows the CUDA libraries come from pip alongside CTranslate2 and are
    not this module's business, so the answer is always yes.
    """
    if sys.platform != "win32":
        return True
    for name in _REQUIRED_DLLS:
        try:
            ctypes.WinDLL(name)
        except OSError:
            return False
    return True


def _download(
    wheel: Wheel, part: Path, log: logging.Logger | None, *, allow_resume: bool = True
) -> str | None:
    """Fetch ``wheel`` into ``part``, resuming a previous attempt when possible.

    Returns the SHA-256 of everything now on disk, or None if the transfer failed.
    Resuming is safe because the caller checks that digest against the pinned one:
    a partial file from a different build, or a corrupted one, simply fails the
    check and is discarded.
    """
    offset = part.stat().st_size if (allow_resume and part.exists()) else 0
    digest = hashlib.sha256()

    if offset:
        if log:
            log.info("resuming %s at %.0f MB", wheel.name, offset / 1e6)
        try:
            with part.open("rb") as existing:
                while chunk := existing.read(_CHUNK):
                    digest.update(chunk)
        except OSError:
            return _download(wheel, part, log, allow_resume=False)

    try:
        with part.open("ab" if offset else "wb") as handle:
            resumed = _stream(
                wheel.url, _HashingWriter(handle, digest), wheel.size, log, offset
            )
    except HTTPError as exc:
        # 416 means the server considers our partial file complete or too long.
        if offset and exc.code == 416:
            part.unlink(missing_ok=True)
            return _download(wheel, part, log, allow_resume=False)
        if log:
            log.warning("downloading %s failed: %s", wheel.name, exc)
        return None
    except OSError as exc:
        if log:
            log.warning("downloading %s failed: %s", wheel.name, exc)
        return None

    if offset and not resumed:
        # The server ignored the range and would have sent the whole file again.
        part.unlink(missing_ok=True)
        return _download(wheel, part, log, allow_resume=False)

    return digest.hexdigest()


def _fetch(wheel: Wheel, target: Path, log: logging.Logger | None) -> bool:
    """Download one wheel, verify it, extract its DLLs, and clean up."""
    target.mkdir(parents=True, exist_ok=True)
    part = target / f"{wheel.name}.part"

    checksum = _download(wheel, part, log)
    if checksum is None:
        part.unlink(missing_ok=True)
        return False

    if checksum != wheel.sha256:
        if log:
            log.error(
                "%s failed its checksum (expected %s, got %s)",
                wheel.name,
                wheel.sha256,
                checksum,
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


def _stream(
    url: str, handle, size: int, log: logging.Logger | None, offset: int = 0
) -> bool:
    """Copy ``url`` into ``handle``, printing progress to the console.

    Returns False, having written nothing, when a range was requested and the
    server answered with the whole file instead.
    """
    request = Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urlopen(request) as response:  # noqa: S310 - a pinned https URL
        if offset and response.status != 206:
            return False
        total = int(response.headers.get("Content-Length") or 0) + offset or size
        done = offset
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                try:
                    print(
                        f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB"
                        f"  ({done * 100 // total}%)",
                        end="",
                        flush=True,
                    )
                except OSError:
                    pass
    say("")
    return True


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

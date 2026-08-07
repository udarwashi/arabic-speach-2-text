"""Audio validation and normalisation.

Whisper wants 16 kHz mono PCM. Rather than trust whatever the browser uploaded,
every file is probed with ``ffprobe`` and then re-encoded with ``ffmpeg``, which
is also what makes MP3, M4A, OGG, FLAC, WebM and MP4 all work the same way.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000

# Extensions accepted from the upload form. ffmpeg handles far more, but an
# explicit list gives the user a clear error instead of a confusing decode failure.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".ogg",
        ".oga",
        ".opus",
        ".flac",
        ".webm",
        ".mp4",
        ".mkv",
        ".mov",
        ".wma",
        ".amr",
        ".aiff",
        ".aif",
    }
)

_PROBE_TIMEOUT = 60
_CONVERT_TIMEOUT = 60 * 30


class AudioError(Exception):
    """Raised when a file cannot be validated or converted.

    ``message`` is safe to show the user (Arabic); ``detail`` carries the tool
    output for logs and for the API's error payload.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class AudioInfo:
    """What ffprobe could tell us about a file."""

    duration: float | None
    codec: str | None
    sample_rate: int | None
    channels: int | None


def is_supported_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def require_ffmpeg() -> None:
    """Fail early, with an actionable message, if the ffmpeg tools are missing."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise AudioError(
                "الأداة ffmpeg غير مثبّتة على النظام.",
                f"{tool} was not found on PATH; install ffmpeg and retry",
            )


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - guarded by require_ffmpeg
        raise AudioError("الأداة ffmpeg غير مثبّتة على النظام.", str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(
            "استغرقت معالجة الملف وقتاً أطول من المسموح.",
            f"{cmd[0]} timed out after {timeout}s",
        ) from exc


def _tail(text: str | None, lines: int = 6) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def probe(path: Path) -> AudioInfo:
    """Read stream metadata, and confirm the file really contains audio."""
    require_ffmpeg()
    result = _run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "a:0",
            str(path),
        ],
        _PROBE_TIMEOUT,
    )
    if result.returncode != 0:
        raise AudioError(
            "تعذّر قراءة الملف الصوتي؛ قد يكون تالفاً أو بصيغة غير مدعومة.",
            _tail(result.stderr),
        )

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioError("تعذّر قراءة بيانات الملف الصوتي.", str(exc)) from exc

    streams = payload.get("streams") or []
    if not streams:
        raise AudioError(
            "لا يحتوي هذا الملف على مسار صوتي.",
            "ffprobe found no audio stream",
        )

    stream = streams[0]
    duration = _first_float(stream.get("duration"), (payload.get("format") or {}).get("duration"))
    if duration is not None and duration <= 0:
        duration = None

    return AudioInfo(
        duration=duration,
        codec=stream.get("codec_name"),
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
    )


def convert_to_wav(source: Path, destination: Path) -> AudioInfo:
    """Re-encode ``source`` to 16 kHz mono 16-bit WAV at ``destination``.

    Returns the info for the *converted* file, whose duration is always known.
    """
    require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(source),
            "-vn",                        # drop any video track (mp4/webm/mkv)
            "-map", "0:a:0",
            "-ac", "1",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(destination),
        ],
        _CONVERT_TIMEOUT,
    )
    if result.returncode != 0 or not destination.exists():
        raise AudioError(
            "فشل تحويل الملف الصوتي إلى الصيغة المطلوبة.",
            _tail(result.stderr) or f"ffmpeg exited with {result.returncode}",
        )

    info = probe(destination)
    if info.duration is None:
        raise AudioError(
            "الملف الصوتي فارغ أو لا يمكن تحديد مدته.",
            "converted file has no measurable duration",
        )
    return info


def _first_float(*values: object) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

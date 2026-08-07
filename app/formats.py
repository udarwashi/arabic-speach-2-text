"""Transcript serialisation: plain text, SRT and WebVTT.

Pure functions over ``Segment`` lists — no I/O, no app state.

Every Arabic line is prefixed with U+200F (RIGHT-TO-LEFT MARK). Subtitle players
and plain-text editors decide paragraph direction from the first strongly-directed
character in the line, so a line that happens to start with a digit or a Latin
word would otherwise be laid out left-to-right.
"""

from __future__ import annotations

from .transcriber import Segment

RLM = "‏"


def _clock(seconds: float, *, millis_separator: str) -> str:
    """Format seconds as ``HH:MM:SS<sep>mmm``, clamping negatives to zero."""
    if seconds < 0 or seconds != seconds:  # negative or NaN
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_separator}{millis:03d}"


def srt_timestamp(seconds: float) -> str:
    return _clock(seconds, millis_separator=",")


def vtt_timestamp(seconds: float) -> str:
    return _clock(seconds, millis_separator=".")


def full_text(segments: list[Segment]) -> str:
    """One flowing paragraph, for the UI and for copy-to-clipboard."""
    return " ".join(segment.text.strip() for segment in _kept(segments))


def to_txt(segments: list[Segment], *, rtl_marks: bool = True) -> str:
    """One segment per line, so the text stays readable without timestamps."""
    prefix = RLM if rtl_marks else ""
    lines = [f"{prefix}{segment.text.strip()}" for segment in _kept(segments)]
    return "\n".join(lines) + ("\n" if lines else "")


def to_srt(segments: list[Segment], *, rtl_marks: bool = True) -> str:
    prefix = RLM if rtl_marks else ""
    blocks: list[str] = []
    for number, segment in enumerate(_ordered(segments), start=1):
        blocks.append(
            f"{number}\n"
            f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}\n"
            f"{prefix}{segment.text.strip()}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[Segment], *, rtl_marks: bool = True) -> str:
    prefix = RLM if rtl_marks else ""
    blocks = ["WEBVTT\n"]
    for segment in _ordered(segments):
        blocks.append(
            f"{vtt_timestamp(segment.start)} --> {vtt_timestamp(segment.end)}\n"
            f"{prefix}{segment.text.strip()}\n"
        )
    return "\n".join(blocks)


FORMATTERS = {
    "txt": (to_txt, "text/plain; charset=utf-8"),
    "srt": (to_srt, "application/x-subrip; charset=utf-8"),
    "vtt": (to_vtt, "text/vtt; charset=utf-8"),
}


def render(fmt: str, segments: list[Segment]) -> tuple[str, str]:
    """Return ``(body, content_type)`` for one of ``FORMATTERS``."""
    try:
        formatter, content_type = FORMATTERS[fmt]
    except KeyError:
        raise ValueError(f"unsupported format {fmt!r}") from None
    return formatter(segments), content_type


def _kept(segments: list[Segment]) -> list[Segment]:
    """Segments that carry actual text. Whitespace-only is not text."""
    return [segment for segment in segments if segment.text and segment.text.strip()]


def _ordered(segments: list[Segment]) -> list[Segment]:
    """Cue order must be monotonic; VAD can emit a segment slightly out of order."""
    return sorted(_kept(segments), key=lambda segment: (segment.start, segment.end))

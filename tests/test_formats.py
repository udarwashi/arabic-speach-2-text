"""Transcript serialisation, including the RTL marks and timestamp arithmetic."""

from __future__ import annotations

import pytest

from app.formats import (
    RLM,
    full_text,
    render,
    srt_timestamp,
    to_srt,
    to_txt,
    to_vtt,
    vtt_timestamp,
)
from app.transcriber import Segment

SEGMENTS = [
    Segment(index=1, start=0.0, end=2.5, text="السلام عليكم"),
    Segment(index=2, start=2.5, end=5.25, text="هذا اختبار للنص العربي"),
]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (59.999, "00:00:59,999"),
        (60.0, "00:01:00,000"),
        (3661.5, "01:01:01,500"),
        (36000.0, "10:00:00,000"),
        (-3.0, "00:00:00,000"),  # clamped rather than emitting a negative cue
    ],
)
def test_srt_timestamp(seconds: float, expected: str) -> None:
    assert srt_timestamp(seconds) == expected


def test_vtt_uses_a_dot_separator() -> None:
    assert vtt_timestamp(1.5) == "00:00:01.500"


def test_millisecond_rounding_does_not_leak_into_seconds() -> None:
    # 0.9999 s must not render as ...:00,1000
    assert srt_timestamp(0.9999) == "00:00:01,000"


def test_full_text_joins_with_single_spaces() -> None:
    assert full_text(SEGMENTS) == "السلام عليكم هذا اختبار للنص العربي"


def test_full_text_ignores_empty_segments() -> None:
    segments = [*SEGMENTS, Segment(index=3, start=5.0, end=6.0, text="")]
    assert full_text(segments) == full_text(SEGMENTS)


def test_txt_puts_each_segment_on_its_own_rtl_marked_line() -> None:
    body = to_txt(SEGMENTS)
    lines = body.rstrip("\n").split("\n")
    assert len(lines) == 2
    assert all(line.startswith(RLM) for line in lines)
    assert lines[0] == f"{RLM}السلام عليكم"


def test_txt_can_omit_rtl_marks() -> None:
    assert RLM not in to_txt(SEGMENTS, rtl_marks=False)


def test_txt_of_nothing_is_empty_not_a_stray_newline() -> None:
    assert to_txt([]) == ""


def test_srt_structure() -> None:
    body = to_srt(SEGMENTS)
    blocks = body.strip().split("\n\n")
    assert len(blocks) == 2

    first = blocks[0].split("\n")
    assert first[0] == "1"
    assert first[1] == "00:00:00,000 --> 00:00:02,500"
    assert first[2] == f"{RLM}السلام عليكم"

    assert blocks[1].split("\n")[0] == "2"


def test_srt_renumbers_from_one_regardless_of_segment_index() -> None:
    segments = [
        Segment(index=7, start=1.0, end=2.0, text="أ"),
        Segment(index=9, start=2.0, end=3.0, text="ب"),
    ]
    numbers = [block.split("\n")[0] for block in to_srt(segments).strip().split("\n\n")]
    assert numbers == ["1", "2"]


def test_cues_are_sorted_by_start_time() -> None:
    unordered = [
        Segment(index=1, start=5.0, end=6.0, text="ثانٍ"),
        Segment(index=2, start=1.0, end=2.0, text="أول"),
    ]
    body = to_srt(unordered)
    assert body.index("أول") < body.index("ثانٍ")


def test_blank_segments_produce_no_cue_and_no_line() -> None:
    segments = [*SEGMENTS, Segment(index=3, start=6.0, end=7.0, text="   ")]
    assert len(to_srt(segments).strip().split("\n\n")) == 2
    assert len(to_txt(segments).rstrip("\n").split("\n")) == 2
    assert full_text(segments) == full_text(SEGMENTS)


def test_surrounding_whitespace_is_trimmed_from_cues() -> None:
    padded = [Segment(index=1, start=0.0, end=1.0, text="  نص  ")]
    assert to_srt(padded).splitlines()[2] == f"{RLM}نص"
    assert full_text(padded) == "نص"


def test_vtt_starts_with_the_signature() -> None:
    body = to_vtt(SEGMENTS)
    assert body.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500" in body
    assert f"{RLM}السلام عليكم" in body


def test_render_returns_body_and_content_type() -> None:
    body, content_type = render("srt", SEGMENTS)
    assert content_type.startswith("application/x-subrip")
    assert body.startswith("1\n")


def test_render_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported format"):
        render("docx", SEGMENTS)

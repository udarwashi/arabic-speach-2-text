"""ffprobe/ffmpeg wrappers, exercised against real generated audio."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app.audio import (
    TARGET_SAMPLE_RATE,
    AudioError,
    convert_to_wav,
    is_supported_extension,
    probe,
)
from tests.conftest import write_tone


@pytest.mark.parametrize(
    "name", ["a.mp3", "A.MP3", "clip.m4a", "voice.ogg", "x.flac", "movie.mp4", "s.opus"]
)
def test_supported_extensions(name: str) -> None:
    assert is_supported_extension(name)


@pytest.mark.parametrize("name", ["notes.txt", "archive.zip", "noextension", "a.pdf"])
def test_unsupported_extensions(name: str) -> None:
    assert not is_supported_extension(name)


def test_probe_reads_duration_and_channels(tone_wav: Path) -> None:
    info = probe(tone_wav)
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert info.channels == 1
    assert info.sample_rate == 22_050


def test_convert_produces_16k_mono_pcm(tmp_path: Path, tone_wav: Path) -> None:
    destination = tmp_path / "out.wav"
    info = convert_to_wav(tone_wav, destination)

    assert destination.exists()
    assert info.sample_rate == TARGET_SAMPLE_RATE
    assert info.channels == 1
    assert info.duration == pytest.approx(1.0, abs=0.05)

    with wave.open(str(destination), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == TARGET_SAMPLE_RATE
        assert handle.getsampwidth() == 2


def test_convert_downmixes_stereo(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\x00\x10\x00\xf0" * 44_100)

    info = convert_to_wav(stereo, tmp_path / "mono.wav")
    assert info.channels == 1
    assert info.sample_rate == TARGET_SAMPLE_RATE


def test_probe_rejects_a_file_that_is_not_audio(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp3"
    fake.write_bytes(b"this is not audio, no matter what the extension says")
    with pytest.raises(AudioError) as excinfo:
        probe(fake)
    assert excinfo.value.message  # a user-facing Arabic message is always present


def test_probe_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        probe(tmp_path / "nope.mp3")


def test_convert_reports_failure_with_detail(tmp_path: Path) -> None:
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"\x00" * 32)
    with pytest.raises(AudioError) as excinfo:
        convert_to_wav(fake, tmp_path / "out.wav")
    assert excinfo.value.detail


def test_probe_rejects_a_video_only_file(tmp_path: Path) -> None:
    """A silent MP4 has no audio stream, which must be a clean error."""
    import subprocess

    video = tmp_path / "silent.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-pix_fmt", "yuv420p", str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:  # pragma: no cover - depends on ffmpeg build
        pytest.skip("this ffmpeg build cannot synthesise a test video")

    with pytest.raises(AudioError, match="مسار صوتي"):
        probe(video)


def test_zero_length_audio_is_rejected(tmp_path: Path) -> None:
    empty = write_tone(tmp_path / "empty.wav", seconds=0.0)
    with pytest.raises(AudioError):
        convert_to_wav(empty, tmp_path / "out.wav")

"""Tests for ffmpeg video mux helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from prob_ml.video import mux_frames_to_mp4


def test_mux_frames_skips_when_first_frame_missing(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "v.mp4"
    assert mux_frames_to_mp4(frames, out, fps=30, num_frames=10) is False


def test_mux_frames_calls_ffmpeg_when_pattern_exists(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 4):
        (frames / f"frame_{i:05d}.png").write_bytes(b"\x89PNG")

    out = tmp_path / "v.mp4"
    with patch("prob_ml.video.subprocess.run") as run:
        run.return_value.returncode = 0
        with patch("prob_ml.video.shutil.which", return_value="/bin/ffmpeg"):
            ok = mux_frames_to_mp4(frames, out, fps=30, num_frames=3)
    assert ok is True
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert "libx264" in cmd
    assert str(out) in cmd

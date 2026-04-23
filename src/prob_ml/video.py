"""Encode rendered frame sequences to H.264 MP4 using ffmpeg (optional)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def mux_frames_to_mp4(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: int,
    num_frames: int,
    frame_pattern: str = "frame_%05d.png",
    start_number: int = 1,
    ffmpeg_bin: str = "ffmpeg",
    crf: int = 20,
) -> bool:
    """
    Mux a directory of numbered PNGs into a yuv420p H.264 file.

    Returns True if encoding succeeded, False if ffmpeg is missing or failed.
    """
    def _resolve_ffmpeg(name: str) -> str | None:
        p = Path(name)
        if p.is_file():
            return str(p.resolve())
        return shutil.which(name)

    ffmpeg = _resolve_ffmpeg(ffmpeg_bin)
    if not ffmpeg:
        return False

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    pattern = str(Path(frames_dir) / frame_pattern)
    first = Path(frames_dir) / frame_pattern.replace("%05d", f"{start_number:05d}")
    if not first.is_file():
        return False

    cmd: list[str | Path] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        pattern,
        "-frames:v",
        str(num_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True

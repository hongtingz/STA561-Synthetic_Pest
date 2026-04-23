"""
Mux rendered PNG frames → MP4 (H.264). Use when you already have
output/frames/frame_*.png and want to rebuild the video without re-rendering.

Requires ffmpeg on PATH.

Run:
    python demo/frames_to_video.py

Or:
    python demo/frames_to_video.py /path/to/frames 30 /path/to/out.mp4
"""

import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FRAMES = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output", "frames"))
DEFAULT_OUT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output", "synthetic_pest_video.mp4"))
DEFAULT_FPS = 30
DEFAULT_NUM = 60


def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FRAMES
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FPS
    out_mp4 = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found on PATH. Install: https://ffmpeg.org/ (brew install ffmpeg)")
        sys.exit(1)

    first = os.path.join(frames_dir, "frame_0001.png")
    if not os.path.isfile(first):
        print(f"Missing {first} — run Blender render first.")
        sys.exit(1)

    pattern = os.path.join(frames_dir, "frame_%04d.png")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(fps), "-start_number", "1", "-i", pattern,
        "-frames:v", str(DEFAULT_NUM),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        out_mp4,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {out_mp4}")


if __name__ == "__main__":
    main()

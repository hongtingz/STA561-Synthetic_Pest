"""
Mux rendered PNG frames → MP4 (H.264), same contract as the main pipeline
(`src/prob_ml/video.py` — `frame_%05d.png`).

Run from repo root:
    uv run python demo/frames_to_video.py
    uv run python demo/frames_to_video.py path/to/frames 30 path/to/out.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

from prob_ml.video import mux_frames_to_mp4

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
DEFAULT_FRAMES = REPO / "artifacts" / "render" / "frames"
DEFAULT_OUT = REPO / "artifacts" / "render" / "video.mp4"
DEFAULT_FPS = 30
DEFAULT_NUM = 900  # 30s @ 30fps; override by argv or match your render seconds


def main() -> None:
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAMES
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FPS
    out_mp4 = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT
    num_frames = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_NUM

    first = frames_dir / "frame_00001.png"
    if not first.is_file():
        print(f"Missing {first} — run `pest-pipeline render` with execute=true first.")
        sys.exit(1)

    ok = mux_frames_to_mp4(
        frames_dir,
        out_mp4,
        fps=fps,
        num_frames=num_frames,
        frame_pattern="frame_%05d.png",
        start_number=1,
    )
    if not ok:
        print("Mux failed. Install ffmpeg (e.g. brew install ffmpeg) and ensure frames exist.")
        sys.exit(1)
    print(f"Wrote {out_mp4}")


if __name__ == "__main__":
    main()

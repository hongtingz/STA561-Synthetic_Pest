"""Unified command-line entrypoint for the pest pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys

from prob_ml.config import load_config
from prob_ml.dataset import run_convert
from prob_ml.dcc import build_sbatch_command, validate_job_scripts
from prob_ml.infer import run_infer
from prob_ml.pipeline import ensure_runtime_directories, render_plan
from prob_ml.render import run_render
from prob_ml.train import run_train


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="pest-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--config",
            default="configs/base.json",
            help="Path to the runtime JSON config file.",
        )

    for name in ["plan", "render", "convert", "train", "infer", "pipeline"]:
        add_config_argument(subparsers.add_parser(name))

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument(
        "--config",
        default="configs/base.json",
        help="Path to the runtime JSON config file.",
    )

    dcc_parser = subparsers.add_parser("dcc-submit")
    add_config_argument(dcc_parser)
    dcc_parser.add_argument(
        "--job",
        default="pipeline",
        choices=["pipeline", "render", "train"],
        help="Which Slurm job script to target.",
    )
    return parser


def run_doctor(config_path: str) -> int:
    """Check local prerequisites for the scaffold."""
    config = load_config(config_path)
    print(f"Config: {config.path}")
    for tool in ["uv", "python", "blender", "sbatch"]:
        print(f"{tool}: {'found' if shutil.which(tool) else 'missing'}")
    print("job scripts:")
    for script in validate_job_scripts(config):
        print(f"  - {script.relative_to(config.repo_root)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor(args.config)

    config = load_config(args.config)
    created_dirs = ensure_runtime_directories(config)

    if args.command == "plan":
        print(render_plan(config))
        print("Created directories:")
        for path in created_dirs:
            print(f"  - {path.relative_to(config.repo_root)}")
        return 0

    if args.command == "render":
        run_render(config)
        return 0

    if args.command == "convert":
        run_convert(config)
        return 0

    if args.command == "train":
        run_train(config)
        return 0

    if args.command == "infer":
        run_infer(config)
        return 0

    if args.command == "pipeline":
        print(render_plan(config))
        print("\n== render ==")
        run_render(config)
        print("\n== convert ==")
        run_convert(config)
        print("\n== train ==")
        run_train(config)
        print("\n== infer ==")
        run_infer(config)
        return 0

    if args.command == "dcc-submit":
        print(build_sbatch_command(config, args.job))
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

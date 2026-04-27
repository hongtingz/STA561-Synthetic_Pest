"""Unified command-line entrypoint for the pest pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys

from prob_ml.config import load_config
from prob_ml.dataset import run_convert
from prob_ml.dcc import build_sbatch_command, validate_job_scripts
from prob_ml.evaluate import run_evaluate
from prob_ml.infer import run_infer
from prob_ml.pipeline import ensure_runtime_directories, render_plan
from prob_ml.render import run_render, run_render_batch
from prob_ml.sanity import run_sanity_check
from prob_ml.train import run_train
from prob_ml.yolo import run_train_yolo


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

    for name in [
        "plan",
        "render",
        "render-batch",
        "post-render",
        "convert",
        "sanity-check",
        "train",
        "evaluate",
        "train-yolo",
        "infer",
        "pipeline",
    ]:
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
        choices=[
            "pipeline",
            "post-render",
            "render",
            "render-batch",
            "convert",
            "sanity-check",
            "train",
            "evaluate",
            "train-yolo",
        ],
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


def run_named_stage(command: str, config) -> None:
    """Dispatch a stage name to the corresponding implementation."""
    if command == "render":
        run_render(config)
        return
    if command == "render-batch":
        run_render_batch(config)
        return
    if command == "post-render":
        pipeline_cfg = config.section("pipeline")
        stages = pipeline_cfg.get(
            "post_render_stages",
            ["convert", "sanity-check", "train", "evaluate"],
        )
        if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
            raise TypeError("pipeline.post_render_stages must be a JSON array of command names.")
        for stage in stages:
            run_named_stage(stage, config)
        return
    if command == "convert":
        run_convert(config)
        return
    if command == "sanity-check":
        run_sanity_check(config)
        return
    if command == "train":
        run_train(config)
        return
    if command == "evaluate":
        run_evaluate(config)
        return
    if command == "train-yolo":
        run_train_yolo(config)
        return
    if command == "infer":
        run_infer(config)
        return
    raise ValueError(f"Unsupported pipeline stage: {command}")


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

    if args.command in {
        "render",
        "render-batch",
        "post-render",
        "convert",
        "sanity-check",
        "train",
        "evaluate",
        "train-yolo",
        "infer",
    }:
        run_named_stage(args.command, config)
        return 0

    if args.command == "pipeline":
        print(render_plan(config))
        pipeline_cfg = config.section("pipeline")
        stages = pipeline_cfg.get(
            "stages",
            ["render-batch", "convert", "sanity-check", "train", "evaluate"],
        )
        if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
            raise TypeError("pipeline.stages must be a JSON array of command names.")
        print("\nPipeline stages:")
        for stage in stages:
            print(f"  - {stage}")
        for stage in stages:
            print(f"\n== {stage} ==")
            run_named_stage(stage, config)
        return 0

    if args.command == "dcc-submit":
        print(build_sbatch_command(config, args.job))
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

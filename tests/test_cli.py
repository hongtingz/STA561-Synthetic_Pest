"""Lightweight tests for the rebuilt CLI scaffold."""

from __future__ import annotations

from prob_ml.cli import build_parser


def test_parser_accepts_plan_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["plan", "--config", "configs/base.json"])
    assert args.command == "plan"
    assert args.config == "configs/base.json"


def test_parser_accepts_render_batch_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["render-batch", "--config", "configs/base.json"])
    assert args.command == "render-batch"
    assert args.config == "configs/base.json"


def test_parser_accepts_train_yolo_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-yolo", "--config", "configs/base.json"])
    assert args.command == "train-yolo"
    assert args.config == "configs/base.json"


def test_parser_accepts_sanity_and_evaluate_commands() -> None:
    parser = build_parser()
    sanity_args = parser.parse_args(["sanity-check", "--config", "configs/base.json"])
    eval_args = parser.parse_args(["evaluate", "--config", "configs/base.json"])
    assert sanity_args.command == "sanity-check"
    assert eval_args.command == "evaluate"

"""DCC submission helpers."""

from __future__ import annotations

from pathlib import Path

from prob_ml.config import PipelineConfig


def build_sbatch_command(config: PipelineConfig, job_name: str) -> str:
    """Return the sbatch command for a named job."""
    if job_name == "pipeline":
        helper_script = config.repo_root / "scripts" / "dcc_submit.sh"
        return (
            f"PROJECT_ROOT={config.repo_root} "
            f"bash {helper_script} pipeline {config.path}"
        )
    script_dir = config.section("dcc").get("job_script_dir", "jobs")
    script_path = config.repo_root / script_dir / f"{job_name}.sbatch"
    if not script_path.exists():
        raise FileNotFoundError(f"Job script not found: {script_path}")
    return f"PROJECT_ROOT={config.repo_root} CONFIG_PATH={config.path} sbatch {script_path}"


def validate_job_scripts(config: PipelineConfig) -> list[Path]:
    """Return available sbatch scripts."""
    script_dir = config.repo_root / config.section("dcc").get("job_script_dir", "jobs")
    return sorted(script_dir.glob("*.sbatch"))

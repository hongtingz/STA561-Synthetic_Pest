# Submission Documents

This directory contains the Markdown sources that map directly to the required
submission structure in `PROJECT_SPEC.md`.

Files:

- `executive_summary.md`
  Source for the two-page executive summary.
- `faq.md`
  Source for the 2-5 page FAQ.
- `technical_appendix.md`
  Source for the technical appendix.
- `results/`
  Lightweight current DCC result snapshot: JSON reports plus representative
  images. Large `.pt` checkpoints are intentionally not committed.

Recommended update workflow:

1. Update the Markdown sources when DCC training/evaluation artifacts change.
2. Regenerate the root-level PDFs (`summary.pdf`, `FAQ.pdf`,
   `technical_appendix.pdf`).
3. Refresh `results/` with small JSON reports and selected images if a newer
   DCC run supersedes the current snapshot.
4. Keep generated model checkpoints and bulk rendered frames out of git.

Supporting materials elsewhere in the repository:

- `DCC_DEPLOYMENT.md`
  Instructor-facing DCC reproduction guide (recommended job order, expected
  artifact paths under **Section 8**).
- `CONTRIBUTIONS.md`
  Team contribution split.
- `README.md`
  Data directory conventions, full CLI list, and design notes.
- `pyproject.toml`
  Python version (`>=3.12`) and pinned runtime dependencies.
- `configs/dcc_gpu.json`
  Authoritative defaults for DCC render, training, evaluation, and cluster
  resources.
- `notebooks/dcc_pipeline_demo.ipynb`
  Lightweight notebook demo for reviewing generated DCC artifacts.

The technical appendix documents: the kitchen manifest CSV schema,
module-to-file map, exact default hyperparameters from `dcc_gpu.json`, metric
definitions in code, ViT/YOLOS-tiny training, Faster R-CNN fallback behavior,
and optional `ultralytics` usage.

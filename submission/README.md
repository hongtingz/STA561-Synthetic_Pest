# Submission Documents

This directory contains the prose drafts that map directly to the required
submission structure in `PROJECT_SPEC.md`.

Files:

- `executive_summary.md`
  Draft for the two-page executive summary.
- `faq.md`
  Draft for the 2-5 page FAQ.
- `technical_appendix.md`
  Draft for the technical appendix.

Recommended workflow:

1. Continue editing these Markdown drafts until the content is stable.
2. When the team is ready for final formatting, convert each draft into the
   final LaTeX source owned by the teammate handling typesetting.
3. Export the final submission as PDF.

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

The technical appendix now also documents: the kitchen manifest CSV schema,
module-to-file map, exact default hyperparameters from `dcc_gpu.json`, metric
definitions in code, and the difference between core dependencies and optional
`ultralytics` / planned `transformers` usage.

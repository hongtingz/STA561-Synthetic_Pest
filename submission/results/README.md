# Current Result Snapshot

This directory contains a lightweight snapshot of the current DCC outputs. It is
intended for review and documentation, not for checkpoint storage. The full
local result folder (`prob_ml_results/`) contains large `.pt` checkpoint files
and is intentionally ignored by git.

## Included Reports

- `reports/dataset_summary.json`
  Split sizes, category ids, frame stride, and missing-background diagnostics.
- `reports/dataset_sanity_report.json`
  Dataset integrity checks. Current status: `pass`.
- `reports/training_report.json`
  ViT/YOLOS-tiny training history, checkpoint metadata, final metrics, and
  threshold sweep.
- `reports/detector_evaluation_report.json`
  Evaluation metrics at thresholds 0.30, 0.50, and 0.70 plus failure-example
  references.

## Current DCC Run Summary

- Detector: ViT/YOLOS-tiny (`hustvl/yolos-tiny`)
- Training set: 1,200 rendered frames, 3,600 boxes
- Validation set: 300 rendered frames, 900 boxes
- Real negative holdout: 709 no-pest kitchen images
- Sanity check: pass, with no split leakage, errors, or warnings

At confidence threshold 0.30, the detector reaches 91.44% TDR on the rendered
validation split and 0.00% image-level FPR on that positive validation split.
On the real no-pest holdout, the image-level FPR is 59.66%, showing that the
pipeline is operational but thresholding/domain adaptation still need work for
real-kitchen false-alarm control.

At confidence threshold 0.70, validation TDR remains above the 80% target
(83.33%), while the real no-pest holdout FPR drops to 17.77%. This threshold
sweep is useful evidence for the sensitivity/false-alarm tradeoff, but the
current run does not yet satisfy the <5% real-negative FPR target.

## Included Images

- `images/rendered_frame_00106.png`
  Example rendered frame.
- `images/train_overlay_frame_00106.jpg`
  Training annotation overlay.
- `images/val_overlay_frame_00106.jpg`
  Validation annotation overlay.
- `images/val_false_negative_000.jpg`
  Example validation false-negative visualization.
- `images/neg_test_false_positive_000.jpg`
  Example real-negative false-positive visualization.
- `images/neg_test_overlay_N09m.jpg`
  Example real negative image overlay/sanity visualization.

Large detector checkpoint files are not committed. If a reviewer needs the
trained weights, they should be shared separately through a release asset,
external storage link, or DCC artifact path.

# FAQ

## 1. Why not train directly on real pest images?

That would require a large labeled dataset of mice, rats, and cockroaches in
real kitchens, including bounding boxes. Such a dataset is hard to collect,
hard to annotate, and unlikely to be available at the scale needed for detector
training.

## 2. Why is synthetic data a good fit for this problem?

Synthetic data is especially useful when positive examples are rare and labels
are expensive. In a rendered scene, pest identity and location are known by
construction, so frame-level annotations can be produced automatically. This
makes it possible to generate a large amount of labeled training data at low
cost.

## 3. Does the system reconstruct a full 3D kitchen from one image?

No. The system does not attempt perfect single-image 3D reconstruction.
Instead, it extracts simple visual cues and uses them to generate a plausible
kitchen layout with the same overall perspective and structure. This is a
deliberate tradeoff between realism and practicality.

## 4. Why is an approximate layout acceptable?

For this project, the main objective is to generate useful labeled data for
pest detection, not to reproduce every physical detail of the kitchen. If the
rendered scene captures the viewpoint, major geometry, and plausible pest
placement well enough, it can still provide meaningful training data.

The approximation also makes the system scalable. A perfect reconstruction
pipeline would likely require manual cleanup or specialized depth data, while
our approach can be applied to many kitchen images using the same Python and
Blender workflow. This is important because the project goal is synthetic data
generation at scale, not one manually crafted kitchen scene.

## 5. Why use Blender?

Blender is well suited for programmable scene construction, animation, and
rendering. It also provides a practical path to large-scale synthetic data
generation, especially when paired with Python automation and cluster
execution.

## 6. Why generate videos instead of only still images?

The project specification emphasizes video, but a video is ultimately a
sequence of frames. Treating the output as labeled frame sequences preserves
the spirit of the assignment while still using standard object-detection
formats for training and evaluation.

## 7. How are bounding boxes generated?

The renderer computes the projected image-space extents of each pest object
relative to the Blender camera. Those projected boxes are written automatically
for each rendered frame. This eliminates the need for manual annotation on the
synthetic data.

After rendering, the dataset conversion step validates that each box has a
positive width and height and lies within the image boundary. This sanity check
is important because synthetic annotations are only useful if the geometry,
camera projection, and exported image paths remain consistent.

## 8. Why support three pest classes?

Because the assignment explicitly targets mice, rats, and cockroaches. A system
that only handled one class would not fully satisfy the project specification.

## 9. Why export both COCO and YOLO formats?

COCO is a standard format for object detection research and makes the dataset
contract explicit. YOLO format is convenient for fast detector baselines and
practical training workflows. Exporting both keeps the dataset portable across
model choices. The repository currently includes a ViT/YOLOS-tiny detector path,
a Faster R-CNN fallback, and an optional Ultralytics YOLO runner.

## 10. What is the role of the real kitchen images with no pests?

Those images are valuable as hard negatives. They help evaluate whether the
detector hallucinates pests in realistic kitchen backgrounds. This matters
because a model that performs well only on synthetic scenes may still generate
too many false positives on real images.

These real negative images are also useful for threshold selection. A detector
may find pests on synthetic validation frames but still be too sensitive on
clean kitchens. Evaluating `neg_test` at several confidence thresholds gives a
more practical view of the false-alarm tradeoff.

## 11. Why is there a `neg_test` split instead of a standard positive `test` split?

At present, we do not have real positive pest images. That means real-world
evaluation is strongest on the false-positive side. The `neg_test` split
preserves real no-pest kitchen images for that purpose. If a positive held-out
test split is added later, it will likely come from rendered or composited
data.

## 12. What are the biggest limitations of the current system?

The main limitations are the realism gap between synthetic and real kitchens,
the lack of real positive pest images, and false-positive control on real
negative kitchens. The latest DCC run clears the 80% detection target on
rendered validation data, but the negative-only holdout still has too many
false alarms to claim that the <5% real-kitchen FPR target has been met.

This is a useful outcome rather than only a failure: it tells us that the data
generation, annotation, training, and evaluation loop works, and it identifies
the next modeling priority as hard-negative mining, better threshold
calibration, and improved synthetic realism.

## 13. How does the model address the vision-transformer requirement?

The main DCC config now sets `detector_model` to `vit`, which resolves to the
Hugging Face `hustvl/yolos-tiny` object detector. The training, inference, and
evaluation code detect that model type and use the transformer-specific input
and prediction path. Faster R-CNN remains available as a lightweight fallback,
and YOLO remains available as an optional comparison.

## 14. Is the project reproducible?

Yes. Reproducibility is one of the main design goals. The repository is
organized around a single Python package, JSON config files, Slurm job scripts,
standard output directories, and a DCC-oriented notebook demo that reads
existing pipeline artifacts.

The technical appendix and `DCC_DEPLOYMENT.md` describe the expected command
sequence: render, convert, sanity-check, train, evaluate, and infer. The
repository also includes tests for the manifest parser, dataset conversion,
detector matching metrics, layout generation, video muxing, and CLI behavior.

## 15. How should a reviewer reproduce the raw-data side of the project?

The code expects kitchen images to be referenced by a manifest CSV under
`data/raw/kitchen/metadata/manifest.csv`. The manifest records file paths,
image ids, split labels, and whether each image is enabled for batch rendering.
The repository does not require real pest-positive images; it uses kitchen
backgrounds and generates pest labels synthetically.

For pest geometry, the project can use optional third-party 3D assets whose
source links and attribution are documented in `assets/pests/CREDITS.md`. If
those assets are not present on a reviewer machine or DCC, the renderer falls
back to procedural pest geometry so the code path remains executable.

## 16. Why include dataset sanity checks as a separate step?

Synthetic data can fail in quiet ways: a frame path may be missing, a bounding
box can be outside the image, a class id can be inconsistent, or the same
kitchen background can accidentally appear in both training and validation.
The `sanity-check` command catches these problems before model training.

It also creates annotation overlay images. These overlays are useful for a
human reviewer because they show whether the generated boxes visually align
with the rendered pests. That is often more informative than only reading a
JSON file.

## 17. How are true detection rate and false positive rate measured?

True detection rate is computed by matching predicted boxes to ground-truth
boxes of the same class using an IoU threshold. A ground-truth pest counts as
detected when a prediction with the correct class overlaps it sufficiently.
The code also reports per-class detection rates for mouse, rat, and cockroach.

False positive rate is measured most naturally on `neg_test`, the split of real
kitchen images with no pests. If the detector predicts at least one pest above
the score threshold on such an image, that image counts as a false positive.
The evaluation code sweeps thresholds such as 0.3, 0.5, and 0.7 so the team can
study the tradeoff between sensitivity and false alarms.

## 18. What would improve the project most if more time were available?

The most impactful improvements would be stronger detector training, better
pest assets, more diverse backgrounds and camera conditions, more realistic
evaluation on real positive examples, and direct comparison among ViT/YOLOS-tiny,
Faster R-CNN, and YOLO runs.

## 19. Why keep MobileNet-backed Faster R-CNN if the main config uses ViT?

The Faster R-CNN path is a useful fallback and comparison model. It is small,
fast, and easy to run when the team wants to isolate data-pipeline behavior from
transformer-specific training issues. The main DCC path uses ViT/YOLOS-tiny,
but retaining Faster R-CNN makes the pipeline more robust.

## 20. Why does `pyproject.toml` list `transformers`?

The dependency is used by the ViT/YOLOS-tiny detector path. In the main config,
`detector_model: "vit"` maps to `hustvl/yolos-tiny`, which is loaded through
Hugging Face Transformers.

## 21. Why is YOLO training optional instead of required?

The Ultralytics stack is useful for a second baseline, but it is **not**
declared in the core `pyproject.toml` dependencies to keep the default
environment minimal and avoid extra download weight for teammates who only need
the torchvision path. The CLI command `pest-pipeline train-yolo` checks for
`ultralytics` at runtime and prints install instructions if it is missing.

## 22. What did the current DCC result snapshot show?

The current DCC snapshot contains 1,200 rendered training frames, 300 rendered
validation frames, and 709 real no-pest kitchen images. The sanity report
passes with no split leakage, errors, or warnings.

Using ViT/YOLOS-tiny, the detector reaches 91.44% TDR on rendered validation
data at threshold 0.30 and 83.33% TDR at threshold 0.70. On the real
negative-only holdout, FPR is 59.66% at threshold 0.30 and 17.77% at threshold
0.70. This means the pipeline is complete and trainable, while real-background
false-positive suppression remains the main remaining model challenge.

## 23. How can later DCC results update the repository without changing the design?

The pipeline writes stable report artifacts: `dataset_summary.json`,
`training_report.json`, `detector_evaluation_report.json`, sanity overlays, and
failure-case images. Later runs can update the quantitative tables and notebook
displays by replacing the lightweight files under `submission/results/` or by
copying the generated artifacts into the documented `artifacts/` locations.
The implementation does not need a redesign for those updates.

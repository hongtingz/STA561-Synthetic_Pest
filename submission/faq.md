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

## 8. Why support three pest classes?

Because the assignment explicitly targets mice, rats, and cockroaches. A system
that only handled one class would not fully satisfy the project specification.

## 9. Why export both COCO and YOLO formats?

COCO is a standard format for object detection research and makes the dataset
contract explicit. YOLO format is convenient for fast detector baselines and
practical training workflows. Exporting both keeps the dataset portable across
model choices. The repository currently includes a built-in Faster R-CNN
training path and an optional Ultralytics YOLO runner for a second baseline.

## 10. What is the role of the real kitchen images with no pests?

Those images are valuable as hard negatives. They help evaluate whether the
detector hallucinates pests in realistic kitchen backgrounds. This matters
because a model that performs well only on synthetic scenes may still generate
too many false positives on real images.

## 11. Why is there a `neg_test` split instead of a standard positive `test` split?

At present, we do not have real positive pest images. That means real-world
evaluation is strongest on the false-positive side. The `neg_test` split
preserves real no-pest kitchen images for that purpose. If a positive held-out
test split is added later, it will likely come from rendered or composited
data.

## 12. What are the biggest limitations of the current system?

The main limitations are the realism gap between synthetic and real kitchens,
the lack of real positive pest images, and the fact that final detector
training and evaluation are still being completed. These limitations affect how
strongly we can claim real-world generalization.

## 13. Why use Faster R-CNN first if the course mentions a vision transformer?

The current Faster R-CNN path serves as an engineering baseline. It lets us
verify that rendering, dataset conversion, training, evaluation, and DCC
execution all work end to end. The planned next step is to plug a
transformer-based detector into the same dataset and evaluation workflow rather
than changing the entire pipeline at once.

## 14. Is the project reproducible?

Yes. Reproducibility is one of the main design goals. The repository is
organized around a single Python package, JSON config files, Slurm job scripts,
standard output directories, and a DCC-oriented notebook demo that reads
existing pipeline artifacts.

## 15. What would improve the project most if more time were available?

The most impactful improvements would be stronger detector training, better
pest assets, more diverse backgrounds and camera conditions, more realistic
evaluation on real positive examples, and a transformer-based detector that can
be compared directly against the existing baseline.

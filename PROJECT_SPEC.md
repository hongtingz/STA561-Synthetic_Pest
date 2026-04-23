# Synthetic Data Generation for Pest Detection

Pest detection is critical for preserving hygiene commercial kitchens. A natural approach is to build
a computer vision model for monitoring kitchens (especially during off hours). However, training a
computer vision model for pest detection and identification requires large amounts of labeled data
which (thankfully) is difficult to obtain. An alternative approach is to generate synthetic video
data using a platform such as Blender or Unity. Such data can be generated inexpensively and
the ground truth is known (by construction) without resource intensive human labeling. In this
project, you will create an adaptive synthetic video generator that, given a still image of a kitchen,
generates labeled video data for pest detection (mice, rats, and cockroaches) set in a kitchen with
the same layout as the picture. Videos should be 30-60 seconds in length with each frame being
labeled with the presence of any pests along with their bounding boxes. Video generation must be
automated with python and run a a scale that it would be feasible to generate thousands of videos
on the Duke compute cluster. To receive an A+, write a complete pipeline that takes as input a
kitchen photo, generates the video, and then trains a vision transformer to identify the location
and type of pests. Your model must achieve an 80% true detection rate with a less than 5% false
positive rate on test video data (run by the instructor).

Some things to get you started:

- See blender.org for a series of free tutorials on using blender also search YouTube and Hugging-
Face for a ton of useful links (the following github may also be useful https://github.com/sean-
halpin/synthetic_dataset_creation_blender)
- Remember that videos are just a sequence of images...perhaps best to start with the generation
of static images. (Is video even necessary for this task? Feel free to argue one way or the
other.)
- Omniverse from NVDIA may also be of interest

## What You'll Turn In

- Two-page executive summary explaining the problem, your approach, results, and future
  work. This should be written without mathematical notation or technical jargon.
- An FAQ (2-5 pages) with questions you think an intelligent and skeptical reader might ask
  and your answers.
- A technical appendix (no length requirement). This will be the bulk of your submission. This
  will include all technical details, code, and results. I must be able to reproduce your results
  from only the technical description and links to raw datasets. Where appropriate, include a
  demo in a jupyter notebook.

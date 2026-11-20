# Video landmark workflow

## Manual annotation

1. Extract stills from the demonstration MP4 and retain the generated `frames.csv`. The manifest maps each still filename to its zero-based source frame index and a nominal timestamp.
2. Annotate the same model feature in each selected snapshot using [DataTorch keypoints](https://docs.datatorch.io/tutorial-keypoints.html). Export the annotations in COCO JSON. Inspect numbered overlays and revisit uncertain placements in DataTorch.
3. Re-export after corrections. The current dataset uses source frames **0, 60, 120, 180, 240, 300, 360, and 420**. The correction added the jaw-bone tip at source frame 300.
4. Export the portable landmark table. Unannotated positions have blank coordinate cells.

```bash
python scripts/export_manual.py --annotations Snapshots_fixed-1.json \
  --manifest "Intubation video_frames/frames.csv" --output data/manual_landmarks.csv
```

Coordinates use the original **478 × 850** image grid, with the origin at the top left, x increasing rightwards, and y increasing downwards. The thirteen labels are Bottom Lip, Top Lip, Chin, Jaw, Back of Head (Upper), Chest, Neck (Upper), Back of Head (Lower), Shoulder, Top of Head, Forehead, Neck (Lower), and TMJ. Jaw marks the middle of the bend in the jaw member. TMJ marks its tip. The two back-of-head labels mark separate bends or curves.

The review and comparison scripts accept `--export`, `--frames` and `--output`. They use NumPy with OpenCV for the numbered review and Matplotlib with Pillow for the graphical comparison. The comparison can align the two torso reference points for an exploratory 2D view. The tracker itself uses the unwarped video.

## CoTracker3

[CoTracker3 Offline](https://github.com/facebookresearch/co-tracker) uses visible manual marks as query points. Each interval between manually labelled frames is tracked from both ends by running forward on the original and reversed frame order. The tail after the final manual frame runs forward. Each directional prediction must pass CoTracker’s visibility check and fall inside the original image.

When both directions succeed within **8 px** of each other, the prediction from the closer reference is retained. A larger difference leaves a gap. A single visible prediction is retained with a one-seed status so it can be reviewed separately. This threshold is a review setting chosen for these images. Manual positions take precedence and missing manual annotations remain as recorded at their reference frames. The script saves one row for every frame and landmark.

Validation compares one direction’s prediction at the far end of a segment with a manual mark from that endpoint, which was withheld from that direction’s inputs. Reported error is Euclidean distance in pixels and is conditional on a visible, in-bounds prediction. Coverage and disagreements are reported separately. This check measures cross-snapshot agreement on the source clip. It cannot establish accuracy on every unmarked frame or on other recordings.

## Local run

For the Intel Mac benchmark, use Python 3.11 and install the tracking requirements in a virtual environment. A CUDA-enabled PyTorch environment can use `--device cuda`.

```bash
python3.11 -m venv .venv-cotracker
source .venv-cotracker/bin/activate
python -m pip install -r requirements-tracking.txt
git clone https://github.com/facebookresearch/co-tracker.git .venv-cotracker/upstream
git -C .venv-cotracker/upstream checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
mkdir -p .venv-cotracker/weights
curl -fL https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth \
  -o .venv-cotracker/weights/scaled_offline.pth

python scripts/track_video.py --video "Intubation video.mp4" \
  --manual data/manual_landmarks.csv \
  --upstream .venv-cotracker/upstream \
  --checkpoint .venv-cotracker/weights/scaled_offline.pth \
  --output local_outputs/cotracker
```

Add `--benchmark-only` to stop after the first timed segment, which targets frames 240–300 for this clip. Results use `manual`, `model_two_seeds`, `model_one_seed`, `disagreement`, `no_visible_prediction`, and the original omission statuses. The review video uses filled dots for manual marks and rings for model suggestions. The report records the benchmark conditions, checkpoint hash, validation errors, and run coverage.

# Ghost Gear Detector Training

Standalone code for training and scoring your own object-detection model that
finds ghost gear (lost or abandoned fishing gear) in side-scan sonar. It uses
the GhostVision derelict-crab-pot dataset, independent of the GhostVision app
itself. No GUI, no Roboflow account, no GhostVision package. Just four scripts:
convert the data, train a model, score it, and look at what it does.

This exists to reproduce (as closely as reasonably possible) the training setup
described in the paper this dataset comes from:

> Bodine, C. S. et al. (2026). *GhostVision: Democratizing Derelict Gear Detection
> Using Low-Cost Sonar and Artificial Intelligence.* Journal of Marine Science and
> Engineering, 14(10), 951. https://doi.org/10.3390/jmse14100951 (Table 1 =
> training settings, Table 3 = the test-set numbers we compare against.)

If you haven't touched an ML training pipeline before, that's fine — every
script's docstring and every setting's `--help` text explains what it does and
why, in plain language, not just what to type. Read a script top to bottom before
running it; the comments are written to be read.

## What's here

| File | What it does |
|---|---|
| `dataset_to_yolo.py` | Converts the dataset's raw format into what the training code expects |
| `train.py` | Trains a YOLO model on the converted data |
| `evaluate.py` | Scores a trained model against held-out test images |
| `visualize.py` | Draws the model's detections on test images, plus report-ready charts |
| `environment.yml` | The (small) list of Python packages this needs |
| `slurm/`, `cluster.env.example` | Optional: running the above as batch jobs on a Slurm cluster |

Run them in that order. Each one's `--help` documents every setting:

```
python train.py --help
```

## Install

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate crabpot-training
```

Training wants an NVIDIA GPU. It will run on CPU (`--device cpu`), just very
slowly — fine for checking that everything is wired up, not for a real run.

### Get the dataset

It's on Hugging Face as a *gated* dataset — you'll need a free HF account and to
click "Agree and access repository" on
[the dataset page](https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds)
first. Then download and convert it:

```bash
pip install "huggingface_hub[cli]"
hf auth login
hf download PINGEcosystem/sss-crab-pot-detection-ds --repo-type dataset --local-dir ./sss-crab-pot-detection-ds
python dataset_to_yolo.py --src ./sss-crab-pot-detection-ds --dst ./sss-crab-pot-detection-yolo
```

If someone in your group has already done this on a shared machine, just point
`--data` at their converted `data.yaml` instead.

## Running it

**Train** (this is the slow part — expect anywhere from ~15 minutes to a couple
hours depending on your GPU and whether it stops early):

```bash
python train.py --arch yolo12s --data <dataset>/data.yaml --out runs/yolo12s
```

**Evaluate**, once training finishes:

```bash
python evaluate.py --weights runs/yolo12s/train/weights/best.pt --data <dataset>/data.yaml --split test
```

This prints five numbers (Precision@50, Recall@50, F1@50, mAP@50, mAP@50-95) -
what each one means is explained in `evaluate.py`'s own docstring, worth reading
before you interpret the output.

**Visualize** — look at what the model actually does, not just its scores:

```bash
python visualize.py --weights runs/yolo12s/train/weights/best.pt --data <dataset>/data.yaml --out figures/yolo12s

# or compare models on the same charts:
python visualize.py --weights runs/yolo12s/train/weights/best.pt runs/yolo26s/train/weights/best.pt \
    --names yolo12s yolo26s --data <dataset>/data.yaml --out figures/compare
```

That writes to `figures/...`:

| File | Shows |
|---|---|
| `<model>_examples.png` | Test images with boxes drawn: clean hits, worst misses, worst false alarms |
| `<model>_confusion.png` | Found / missed / false-alarm counts, and any class mix-ups |
| `pr_curve.png` | Precision vs. recall across the whole confidence range |
| `threshold_sweep.png` | Precision, recall, F1 at each confidence cutoff — where 0.25 sits, and what's best |
| `training_curves.png` | Loss and validation mAP per epoch (overfitting shows up here) |
| `summary.csv` | The numbers behind the charts |

The examples figure is the one to look at first: it tells you *what kind* of
mistakes the model makes, which the five numbers can't.

### On a Slurm cluster (optional)

Don't run training directly on a cluster's login node — it's shared, and
long processes get killed when you log out. Submit a batch job instead:

```bash
cp cluster.env.example cluster.env      # once: fill in your cluster's partition, environment, dataset path
source cluster.env                      # every login, before sbatch
sbatch slurm/train.sbatch yolo12s
sbatch slurm/visualize.sbatch figures/yolo12s runs/yolo12s/train/weights/best.pt
```

Watch jobs with `squeue --me`; each writes a `<name>_<jobid>.log` in the repo
folder. To look at the figures on your own computer, copy them down:

```bash
rsync -avz <you>@<cluster-login-host>:<path-to-repo>/figures/ ./figures/
```

## What to expect

Two reference runs — training loss curves both looked
normal (steadily decreasing, no red flags), and both results are real, not
tuned to look good:

| | Precision@50 | Recall@50 | F1@50 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| **Our YOLOv12s** | 0.324 | 0.252 | 0.283 | 0.217 | 0.081 |
| Paper's YOLOv12 | 0.516 | 0.263 | 0.348 | 0.157 | 0.060 |
| **Our YOLOv26s** | 0.496 | 0.399 | 0.442 | 0.402 | 0.161 |
| Paper's YOLOv26 | 0.667 | 0.085 | 0.150 | 0.074 | 0.030 |

Neither matches exactly (see "Known differences from the paper" below for why),
and that's expected, not a failure — the point is training and scoring your own
model, not reproducing a checksum. Recall lines up closely for YOLOv12; YOLOv26
generalizes noticeably better here than in the paper's own test-set report.
Precision@50/Recall@50 above were computed at confidence 0.25 (Ultralytics'
default) — the paper doesn't state what threshold it used for these two numbers,
so treat that specific comparison as informative, not confirmed identical
methodology.

## Known differences from the paper's exact setup

- The paper trained YOLOv12 with Roboflow's own GitHub fork, not the
  `ultralytics` pip package this repo uses — same architecture, different
  implementation, "due to inefficiencies with the Ultralytics version" per the
  paper.
- RF-DETR isn't included here — different framework entirely (`rfdetr` +
  `supervision`, not `ultralytics`). Also the least interesting of the three to
  chase: the paper reports it as functionally unusable at default settings
  (1,514 false positives on the test set).
- Optimizer choice beyond the learning rate isn't stated in the paper; this uses
  AdamW. See the comment on `--optimizer` in `train.py` for why getting this
  right at all took catching a real bug — Ultralytics' "auto" optimizer mode
  silently ignores whatever learning rate you ask for.
- Trained on whatever GPU you have, not the paper's Tesla T4 —
  affects how long training takes, not what accuracy to expect.

## Links

- Paper: https://doi.org/10.3390/jmse14100951
- Dataset: https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds
- Ultralytics docs (the training library): https://docs.ultralytics.com/
- Supervision docs (the scoring library): https://supervision.roboflow.com/
- GhostVision itself (the full app this dataset/paper also powers, not needed
  for anything in this repo): https://github.com/PINGEcosystem/GhostVision

## License

MIT — see [LICENSE](LICENSE). The dataset and the pretrained checkpoints have
their own terms; see their pages linked above.

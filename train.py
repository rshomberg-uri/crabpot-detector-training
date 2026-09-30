'''
Step 2 of 3: train a YOLO detector on the converted dataset.

WHAT "TRAINING" ACTUALLY DOES
The model starts from a checkpoint that's already been trained on ~80,000 everyday
photos, so it already knows general shapes and edges. Training here shows it our
sonar images with the true pot locations marked, over and over, nudging its
internal numbers a little each time to get better at recognizing the specific
acoustic signature of a crab pot (bright return + shadow). This is called
fine-tuning, not training from scratch.

WHERE THE DEFAULT SETTINGS COME FROM
Every default below is a literal value from Table 1 of the GhostVision paper
(Bodine et al. 2026, JMSE, doi.org/10.3390/jmse14100951) - not a
reasonable-sounding guess:

    Model      Checkpoint   Batch  Epochs  Patience  Learning rate
    YOLOv12    yolo12s.pt   40     200     30         1e-4
    YOLOv26    yolo26s.pt   48     200     30         1e-4

WHERE THIS CAN'T MATCH THE PAPER EXACTLY (read before trusting a close-but-not-
identical result as a bug):
  - The paper trained YOLOv12 with Roboflow's own GitHub fork, not the
    `ultralytics` pip package this script uses ("due to inefficiencies with the
    Ultralytics version and to ensure compatibility with Roboflow utilities").
    Same architecture, different implementation.
  - Trained on a Tesla T4 (16GB) in the paper; this runs on whatever GPU you
    have. Changes how long it takes, not what accuracy it should reach.
  - The paper states the learning rate but never names an optimizer (the
    algorithm that turns "how wrong was that guess" into "how do we adjust").
    This defaults to AdamW, the conventional pairing for a fixed rate this low.
    See the --optimizer note below for why that choice matters more than it
    sounds like it should.

USAGE
    python train.py --arch yolo12s --data <dataset>/data.yaml --out runs/yolo12s
    python train.py --arch yolo26s --data <dataset>/data.yaml --out runs/yolo26s
'''

import argparse
import os

TABLE_1 = {
    'yolo12s': dict(checkpoint='yolo12s.pt', batch=40),
    'yolo26s': dict(checkpoint='yolo26s.pt', batch=48),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument('--arch', required=True, choices=sorted(TABLE_1.keys()),
                     help='Which architecture to train. Picks the matching Table 1 batch size below '
                          'automatically (override with --batch if you want a different one).')
    ap.add_argument('--data', required=True,
                     help='Path to data.yaml written by dataset_to_yolo.py.')
    ap.add_argument('--out', required=True,
                     help='Folder for this run\'s output: the trained weights, loss/accuracy curves, '
                          'and example prediction images Ultralytics saves automatically.')

    # --- Table 1 settings: these four together define "how training proceeds" ---
    ap.add_argument('--epochs', type=int, default=200,
                     help='One epoch = one full pass through every training image. Table 1: 200. In '
                          'practice --patience below usually stops it well before 200 is reached.')
    ap.add_argument('--patience', type=int, default=30,
                     help='Stop early if the score on the held-out validation images hasn\'t improved '
                          'in this many consecutive epochs - no point grinding through all 200 if it '
                          'plateaued long before that (this is exactly what happened in the paper: '
                          'YOLOv12\'s best result was at epoch 51 of a possible 200). Table 1: 30.')
    ap.add_argument('--lr0', type=float, default=1e-4,
                     help='Learning rate: how big a correction the model makes after each batch. Too '
                          'high and it overshoots and never settles (like an autopilot with too much '
                          'gain); too low and it crawls. Table 1: 1e-4 (0.0001), a small, careful step '
                          'size appropriate for fine-tuning rather than training from nothing.')
    ap.add_argument('--optimizer', default='AdamW',
                     help='The specific rule used to turn "how wrong was that prediction" into "how do '
                          'we adjust the model." IMPORTANT: Ultralytics\' optimizer="auto" mode looks '
                          'convenient but SILENTLY IGNORES --lr0 and picks its own learning rate instead '
                          '- caught in this project by a real run that landed on lr=0.01 instead of the '
                          'requested 1e-4. Leave this on an explicit optimizer (AdamW by default) or '
                          '--lr0 above does nothing.')

    ap.add_argument('--batch', type=int, default=None,
                     help='How many images get processed together before the model updates itself once. '
                          'A speed/memory setting, not an accuracy target - bigger needs more GPU memory '
                          'but gives smoother updates. Default: the Table 1 value for --arch (40 or 48).')
    ap.add_argument('--imgsz', type=int, default=640,
                     help='Images are resized to this many pixels square before going into the model. '
                          'Paper: 640 (matches the dataset\'s native resolution).')
    ap.add_argument('--device', default=None,
                     help='Which GPU to use, e.g. "0", or "cpu" to force CPU-only (slow, but works with '
                          'no GPU at all). Default: let Ultralytics auto-detect.')
    ap.add_argument('--workers', type=int, default=8,
                     help='Background CPU threads that load/decode images while the GPU is busy training '
                          'on the previous batch. Not a training-quality setting, just throughput - lower '
                          'it if you\'re on a machine with few CPU cores.')
    args = ap.parse_args()

    cfg = TABLE_1[args.arch]
    batch = args.batch if args.batch is not None else cfg['batch']

    from ultralytics import YOLO  # deferred: this import alone pulls in torch, so keep it out of --help

    print('Training {} ({}) | batch={} epochs={} patience={} optimizer={} lr0={} imgsz={}'.format(
        args.arch, cfg['checkpoint'], batch, args.epochs, args.patience, args.optimizer, args.lr0, args.imgsz))

    model = YOLO(cfg['checkpoint'])  # downloads the pretrained starting checkpoint the first time it's run
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        batch=batch,
        imgsz=args.imgsz,
        workers=args.workers,
        # Absolute on purpose: newer Ultralytics quietly nests a *relative* project
        # path under runs/detect/, so --out runs/yolo12s would really land in
        # runs/detect/runs/yolo12s/. An absolute path is used exactly as given.
        project=os.path.abspath(args.out),
        name='train',
        exist_ok=True,
    )
    if args.device is not None:
        train_kwargs['device'] = args.device

    model.train(**train_kwargs)
    print('\nDone. Weights: {}/train/weights/best.pt'.format(args.out))
    print('(also: last.pt = final epoch\'s weights, and results.csv / results.png = the training curves)')


if __name__ == '__main__':
    main()

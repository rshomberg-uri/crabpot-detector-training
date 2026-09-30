'''
Step 3 of 3: score a trained model against images it has never seen.

WHY A SEPARATE "TEST" SPLIT
The dataset is divided into three independent groups of survey transects: train
(what the model actually learns from), val (checked periodically during training
to decide when to stop - see --patience in train.py), and test (never touched
until now). Scoring against train would be like grading a survey crew on sites
they already knew the answer to - it flatters the number without proving the
model generalizes to sonar it's never seen. Only test-set numbers mean anything
for "will this actually work on a new survey."

THE FIVE NUMBERS THIS PRINTS
  Precision@50  Of every box the model predicted, what fraction was a real pot?
                Low precision = calling clutter a pot (false alarms) - the same
                failure mode you already watch for reading shadows by eye.
  Recall@50     Of every real pot in the images, what fraction did the model
                find? Low recall = missed targets.
  F1@50         One number balancing the two above, so a model can't win just by
                flagging everything (inflates recall) or almost nothing (inflates
                precision).
  mAP@50        Instead of one confidence cutoff, this sweeps the model's
                sensitivity dial across its whole range and scores it across that
                entire sweep - like scoring a detector across many gain settings
                instead of judging it at one.
  mAP@50-95     The same idea, but also demanding progressively tighter box
                overlap (not just "found it," but "found it precisely"). Harder
                to score well on than mAP@50.
"@50" in each name refers to the overlap threshold (IoU, Intersection over
Union) used to decide a predicted box "hit" the true one: at least 50% overlap.

This computes all five the same way the paper's own Table 3 did (via the
`supervision` library, IoU 0.5), so the printed numbers sit directly next to it:

    Model    Precision@50  Recall@50  F1@50   mAP@50  mAP@50-95
    YOLOv12  0.516         0.263      0.348   0.157   0.060
    YOLOv26  0.667         0.085      0.150   0.074   0.030
    RF-DETR  0.006         0.979      0.011   0.379   0.148

(The paper used `supervision` v0.27.0; whatever version is installed here may
differ slightly.)

USAGE
    python evaluate.py --weights runs/yolo12s/train/weights/best.pt \
        --data <dataset>/data.yaml --split test
'''

import argparse
import os

import supervision as sv
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='Path to a trained best.pt (written by train.py)')
    ap.add_argument('--data', required=True, help='Path to data.yaml')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                     help='Which portion of the dataset to score against. Always use "test" for a real '
                          'result - see the note above on why train/val numbers are misleading.')

    # Why two different confidence thresholds, not one:
    ap.add_argument('--map-conf', type=float, default=0.001,
                     help='Confidence threshold used ONLY for computing mAP. mAP needs to see every '
                          'prediction, including low-confidence ones, to sweep across the full range - so '
                          'this is deliberately left very low ("almost no filtering"). Do not reuse this '
                          'value for Precision/Recall; see --pr-conf.')
    ap.add_argument('--pr-conf', type=float, default=0.25,
                     help='Confidence threshold for the Precision@50/Recall@50/F1@50 numbers specifically '
                          '- Ultralytics\' own standard default. Unlike mAP, Precision/Recall are measured '
                          'at ONE fixed operating point, not swept. Reusing --map-conf\'s near-zero '
                          'threshold here would flood the count with barely-above-noise "predictions" and '
                          'collapse precision artificially - which is exactly what happened on the first '
                          'run of this script (Precision@50 came out ~0.04 on a model that clearly wasn\'t '
                          'that bad), before the two were split apart.')
    ap.add_argument('--iou', type=float, default=0.5,
                     help='How much a predicted box must overlap the true box to count as a correct '
                          'detection (0-1). Paper: 0.5, i.e. at least half overlap.')
    args = ap.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    base = data_cfg.get('path', os.path.dirname(args.data))
    rel = data_cfg[args.split]  # e.g. "images/test"
    images_dir = os.path.join(base, rel)
    labels_dir = os.path.join(base, rel.replace('images', 'labels'))

    print('Loading {} split: {}'.format(args.split, images_dir))
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=images_dir,
        annotations_directory_path=labels_dir,
        data_yaml_path=args.data,
        show_progress=True,
    )
    print('{} images, {} classes: {}'.format(len(dataset), len(dataset.classes), dataset.classes))

    from ultralytics import YOLO
    model = YOLO(args.weights)

    map_metric = sv.metrics.MeanAveragePrecision()
    precision_metric = sv.metrics.Precision()
    recall_metric = sv.metrics.Recall()

    # Two separate prediction passes at the two thresholds explained above.
    predictions_list_map = []
    predictions_list_pr = []
    targets_list = []
    for image_path, image, ground_truth in dataset:
        result_map = model.predict(image, conf=args.map_conf, verbose=False)[0]
        predictions_list_map.append(sv.Detections.from_ultralytics(result_map))

        result_pr = model.predict(image, conf=args.pr_conf, verbose=False)[0]
        predictions_list_pr.append(sv.Detections.from_ultralytics(result_pr))

        targets_list.append(ground_truth)

    map_result = map_metric.update(predictions_list_map, targets_list).compute()
    precision_result = precision_metric.update(predictions_list_pr, targets_list).compute()
    recall_result = recall_metric.update(predictions_list_pr, targets_list).compute()

    p50 = float(precision_result.precision_at_50)
    r50 = float(recall_result.recall_at_50)
    f1_50 = (2 * p50 * r50 / (p50 + r50)) if (p50 + r50) > 0 else 0.0

    print()
    print('{:<12} {:>12} {:>10} {:>8} {:>8} {:>10}'.format('Model', 'Precision@50', 'Recall@50', 'F1@50', 'mAP@50', 'mAP@50-95'))
    print('{:<12} {:>12.3f} {:>10.3f} {:>8.3f} {:>8.3f} {:>10.3f}'.format(
        os.path.basename(args.weights), p50, r50, f1_50, float(map_result.map50), float(map_result.map50_95)
    ))
    print()
    print('Paper Table 3, for reference:')
    print('YOLOv12  0.516  0.263  0.348  0.157  0.060')
    print('YOLOv26  0.667  0.085  0.150  0.074  0.030')
    print('RF-DETR  0.006  0.979  0.011  0.379  0.148')


if __name__ == '__main__':
    main()

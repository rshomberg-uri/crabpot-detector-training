'''
Optional step 4: turn a trained model into pictures you can actually look at.

WHY BOTHER
evaluate.py boils a model down to five numbers. Those numbers can't tell you
*what kind* of mistakes the model makes - does it miss faint pots near the
nadir? Call every rock a pot? Mix up "Crab-Pot" and "Maybe-Crab-Pot"? The only
way to find out is to look at its predictions drawn on real sonar images, the
same way you'd check an operator's picks. This script makes those pictures, plus
the standard charts people expect to see in a report.

Ultralytics already saves some of this during training (results.png,
confusion_matrix.png, val_batch*_pred.jpg in the run folder) - but only on the
*validation* split, which was used to pick the best epoch. Everything here is
computed on the *test* split by default, so it matches evaluate.py.

WHAT IT WRITES (into --out)
  training_curves.png     Loss and validation mAP per epoch, read from the
                          results.csv Ultralytics saved next to the weights.
                          Healthy: both losses fall, val loss flattens rather
                          than climbing back up (climbing = overfitting).
  pr_curve.png            Precision vs. recall as the confidence dial sweeps from
                          "report everything" to "report only sure things."
                          Up-and-to-the-right is better.
  threshold_sweep.png     Precision, recall, and F1 at each confidence setting.
                          Shows where the 0.25 default sits, and which setting
                          would give the best F1 instead.
  <name>_confusion.png    For every true object: did the model find it, and what
                          did it call it? Plus false alarms on empty seafloor.
  <name>_examples.png     Real test images, one row each of: clean hits, the
                          worst misses, and the worst false alarms.
  summary.csv             The numbers behind the charts.

HOW BOXES ARE COUNTED AS RIGHT OR WRONG
A prediction "hits" a true pot if the boxes overlap by at least --iou (0.5 =
half, same as evaluate.py). The confusion matrix and example images match boxes
regardless of class, so a pot labeled "Maybe" still counts as found, and the
matrix then shows the class mix-up. The PR curve and threshold sweep require the
class to match too, pooled over both classes - so its numbers land close to, but
not exactly on, evaluate.py's (supervision averages per class instead).

USAGE
    python visualize.py --weights runs/yolo12s/train/weights/best.pt \\
        --data <dataset>/data.yaml --out figures/yolo12s

    # Compare models on the same charts (--names labels them):
    python visualize.py --weights runs/yolo12s/train/weights/best.pt runs/yolo26s/train/weights/best.pt \\
        --names yolo12s yolo26s --data <dataset>/data.yaml --out figures/compare
'''

import argparse
import csv
import os
import sys

import numpy as np
import yaml

# First three slots of a colorblind-checked categorical palette - one color per
# model, always in this order, so "yolo12s" keeps its color in every chart.
MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']

# Box colors drawn on the sonar images, one per outcome (not per class).
HIT_COLOR = '#1baf7a'     # prediction that matched a true pot
FALSE_COLOR = '#e34948'   # prediction with no true pot under it
MISS_COLOR = '#eda100'    # true pot the model never found
TRUTH_COLOR = '#ffffff'   # true pot that WAS found (thin, so you can compare fit)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def load_split(data_yaml, split):
    '''Return (class_names, [(image_path, gt_classes, gt_boxes_normalized_xywh), ...]).'''
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    base = cfg.get('path', os.path.dirname(os.path.abspath(data_yaml)))
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), base)
    images_dir = os.path.join(base, cfg[split])
    labels_dir = os.path.join(base, cfg[split].replace('images', 'labels'))

    names = cfg['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    items = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + '.txt')
        rows = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                rows = [line.split() for line in f if line.strip()]
        cls = np.array([int(r[0]) for r in rows], dtype=int)
        xywh = np.array([[float(v) for v in r[1:5]] for r in rows], dtype=float).reshape(-1, 4)
        items.append((os.path.join(images_dir, fname), cls, xywh))
    return names, items


def xywhn_to_xyxy(xywh, w, h):
    '''YOLO label format (normalized center + size) -> pixel corners.'''
    cx, cy, bw, bh = xywh[:, 0] * w, xywh[:, 1] * h, xywh[:, 2] * w, xywh[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def iou_matrix(a, b):
    '''Overlap (intersection / union) between every box in a and every box in b.'''
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_any_class(gt_boxes, pred_boxes, iou_thr):
    '''Pair predictions with true boxes, best overlap first, ignoring class.
    Returns (pairs [(gt_i, pred_j)], unmatched_gt, unmatched_pred).'''
    ious = iou_matrix(gt_boxes, pred_boxes)
    pairs, used_g, used_p = [], set(), set()
    for flat in np.argsort(-ious, axis=None):
        g, p = np.unravel_index(flat, ious.shape)
        if ious[g, p] < iou_thr:
            break
        if g in used_g or p in used_p:
            continue
        pairs.append((g, p))
        used_g.add(g)
        used_p.add(p)
    return (pairs,
            [g for g in range(len(gt_boxes)) if g not in used_g],
            [p for p in range(len(pred_boxes)) if p not in used_p])


def tp_flags_same_class(gt_cls, gt_boxes, pred_cls, pred_conf, pred_boxes, iou_thr):
    '''Standard detection scoring: most-confident prediction first, each claims the
    best still-unclaimed true box of the SAME class. Returns a True/False per prediction.'''
    tp = np.zeros(len(pred_conf), dtype=bool)
    ious = iou_matrix(pred_boxes, gt_boxes)
    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for j in np.argsort(-pred_conf):
        if len(gt_boxes) == 0:
            break
        cand = ious[j] * (gt_cls == pred_cls[j]) * (~claimed)
        g = int(np.argmax(cand))
        if cand[g] >= iou_thr:
            tp[j] = True
            claimed[g] = True
    return tp


def predict_all(weights, items, conf, batch):
    from ultralytics import YOLO  # deferred: pulls in torch
    model = YOLO(weights)
    preds = []
    for i in range(0, len(items), batch):
        chunk = [path for path, _, _ in items[i:i + batch]]
        for r in model.predict(chunk, conf=conf, verbose=False):
            b = r.boxes
            preds.append(dict(
                shape=r.orig_shape,  # (height, width)
                boxes=b.xyxy.cpu().numpy(),
                conf=b.conf.cpu().numpy(),
                cls=b.cls.cpu().numpy().astype(int),
            ))
        if sys.stdout.isatty():  # live counter in a terminal; skip it in batch-job logs
            print('  predicted {}/{}'.format(min(i + batch, len(items)), len(items)), end='\r')
    print('  predicted {} images'.format(len(items)))
    return preds


def score_model(items, preds, n_classes, iou_thr, pr_conf):
    '''Everything the charts need, computed once per model.'''
    all_conf, all_tp, n_gt = [], [], 0
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # last row/col = background
    per_image = []
    for (path, gt_cls, gt_xywh), p in zip(items, preds):
        h, w = p['shape']
        gt_boxes = xywhn_to_xyxy(gt_xywh, w, h)
        n_gt += len(gt_boxes)

        all_conf.append(p['conf'])
        all_tp.append(tp_flags_same_class(gt_cls, gt_boxes, p['cls'], p['conf'], p['boxes'], iou_thr))

        keep = p['conf'] >= pr_conf
        boxes, cls = p['boxes'][keep], p['cls'][keep]
        pairs, miss, false = match_any_class(gt_boxes, boxes, iou_thr)
        for g, j in pairs:
            confusion[gt_cls[g], cls[j]] += 1
        for g in miss:
            confusion[gt_cls[g], n_classes] += 1
        for j in false:
            confusion[n_classes, cls[j]] += 1
        per_image.append(dict(path=path, gt_cls=gt_cls, gt_boxes=gt_boxes, boxes=boxes, cls=cls,
                              conf=p['conf'][keep], pairs=pairs, miss=miss, false=false))

    conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    order = np.argsort(-conf)
    conf, tp = conf[order], tp[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(~tp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / max(n_gt, 1)

    thresholds = np.linspace(0.01, 0.95, 95)
    sweep = []
    for t in thresholds:
        k = int(np.searchsorted(-conf, -t, side='right'))  # predictions with conf >= t
        P = cum_tp[k - 1] / k if k else np.nan  # no detections left: precision is undefined, not 1
        R = cum_tp[k - 1] / max(n_gt, 1) if k else 0.0
        F = 2 * P * R / (P + R) if k and (P + R) else 0.0
        sweep.append((t, P, R, F))
    return dict(precision=precision, recall=recall, sweep=np.array(sweep),
                confusion=confusion, per_image=per_image, n_gt=n_gt)


def style_axes(ax):
    ax.grid(True, color='#e4e3df', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#9a9993')
    ax.tick_params(colors='#52514e')


def plot_training_curves(models, out_path):
    import matplotlib.pyplot as plt
    rows = []
    for name, weights, color in models:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(weights))), 'results.csv')
        if not os.path.exists(csv_path):
            print('  (no results.csv next to {} - skipping its training curves)'.format(weights))
            continue
        with open(csv_path) as f:
            data = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        rows.append((name, color, data))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color, data in rows:
        ep = [d['epoch'] for d in data]
        axes[0].plot(ep, [d['train/box_loss'] for d in data], color=color, lw=2, label='{} train'.format(name))
        axes[0].plot(ep, [d['val/box_loss'] for d in data], color=color, lw=2, ls='--', label='{} val'.format(name))
        m95 = [d['metrics/mAP50-95(B)'] for d in data]
        axes[1].plot(ep, [d['metrics/mAP50(B)'] for d in data], color=color, lw=2, label='{} mAP@50'.format(name))
        axes[1].plot(ep, m95, color=color, lw=2, ls='--', label='{} mAP@50-95'.format(name))
        best = int(np.argmax(m95))
        axes[1].plot(ep[best], m95[best], 'o', ms=8, color=color, mec='white', mew=2)
        axes[1].annotate('best epoch {}'.format(int(ep[best])), (ep[best], m95[best]),
                         textcoords='offset points', xytext=(0, -18), ha='center', fontsize=9, color='#52514e',
                         bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    axes[0].set_title('Box loss (lower is better)', loc='left')
    axes[1].set_title('Validation accuracy (higher is better)', loc='left')
    for ax in axes:
        style_axes(ax)
        ax.set_xlabel('epoch')
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_and_sweep(scored, pr_conf, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, color, s in scored:
        ax.plot(s['recall'], s['precision'], color=color, lw=2, label=name)
        t, P, R, F = min(s['sweep'], key=lambda r: abs(r[0] - pr_conf))
        ax.plot(R, P, 'o', ms=8, color=color, mec='white', mew=2)
        ax.annotate('conf {:.2f}'.format(pr_conf), (R, P), textcoords='offset points',
                    xytext=(8, 4), fontsize=9, color='#52514e')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall  (fraction of real pots found)')
    ax.set_ylabel('Precision  (fraction of detections that are real)')
    ax.set_title('Precision-recall on test set', loc='left')
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=150)
    plt.close(fig)

    # One panel per metric (not per model) so each panel has one y-axis meaning.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for name, color, s in scored:
        sw = s['sweep']
        for ax, col, title in zip(axes, (1, 2, 3), ('Precision', 'Recall', 'F1')):
            ax.plot(sw[:, 0], sw[:, col], color=color, lw=2, label=name)
        best = int(np.argmax(sw[:, 3]))
        axes[2].plot(sw[best, 0], sw[best, 3], 'o', ms=8, color=color, mec='white', mew=2)
        axes[2].annotate('best F1 at {:.2f}'.format(sw[best, 0]), (sw[best, 0], sw[best, 3]),
                         textcoords='offset points', xytext=(6, 6), fontsize=9, color='#52514e')
    for ax, title in zip(axes, ('Precision', 'Recall', 'F1')):
        ax.axvline(pr_conf, color='#9a9993', lw=1, ls=':')
        ax.set_title(title, loc='left')
        ax.set_xlabel('confidence threshold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style_axes(ax)
    axes[0].text(pr_conf, 0.97, ' default {:.2f}'.format(pr_conf), fontsize=9, color='#52514e', va='top')
    axes[0].legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'threshold_sweep.png'), dpi=150)
    plt.close(fig)


def plot_confusion(name, s, class_names, pr_conf, out_path):
    import matplotlib.pyplot as plt
    labels = list(class_names) + ['(nothing)']
    m = s['confusion'].astype(float)
    shown = m.copy()
    shown[-1, -1] = np.nan  # "no object, no prediction" isn't countable
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(shown, cmap='Blues', vmin=0, vmax=np.nanmax(shown) or 1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == len(labels) - 1 and j == len(labels) - 1:
                continue
            v = int(m[i, j])
            dark = v > 0.55 * (np.nanmax(shown) or 1)
            ax.text(j, i, str(v), ha='center', va='center', fontsize=12,
                    color='white' if dark else '#0b0b0b')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('What the model said')
    ax.set_ylabel('What was really there')
    ax.set_title('{}: test set, conf >= {:.2f}'.format(name, pr_conf), loc='left')
    ax.text(0, -0.18, 'Right column = missed. Bottom row = false alarms.',
            transform=ax.transAxes, fontsize=9, color='#52514e')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_examples(name, s, class_names, n, out_path):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    imgs = s['per_image']
    clean = sorted([d for d in imgs if d['pairs'] and not d['miss'] and not d['false']],
                   key=lambda d: -len(d['pairs']))
    missed = sorted([d for d in imgs if d['miss']], key=lambda d: (-len(d['miss']), len(d['false'])))
    false = sorted([d for d in imgs if d['false']], key=lambda d: (-len(d['false']), len(d['miss'])))

    # The dataset contains several exported copies of some sonar frames (same
    # name up to ".rf.<hash>"). Show each frame at most once across the figure.
    used = set()

    def pick(pool):
        chosen = []
        for d in pool:
            frame = os.path.basename(d['path']).split('.rf.')[0]
            if frame not in used and len(chosen) < n:
                used.add(frame)
                chosen.append(d)
        return chosen

    groups = [('Clean hits', pick(clean)), ('Worst misses', pick(missed)), ('Worst false alarms', pick(false))]

    short = {i: ('Pot' if 'maybe' not in c.lower() else 'Maybe') for i, c in enumerate(class_names)}
    fig, axes = plt.subplots(len(groups), n, figsize=(4 * n, 4.4 * len(groups)), squeeze=False)
    for row, (title, pool) in enumerate(groups):
        for col in range(n):
            ax = axes[row, col]
            ax.axis('off')
            if col >= len(pool):
                continue
            d = pool[col]
            img = cv2.cvtColor(cv2.imread(d['path']), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            matched_gt = {g for g, _ in d['pairs']}
            matched_pred = {j for _, j in d['pairs']}
            for g, box in enumerate(d['gt_boxes']):
                hit = g in matched_gt
                ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False,
                                       ec=TRUTH_COLOR if hit else MISS_COLOR,
                                       lw=1 if hit else 2, ls='--'))
            for j, box in enumerate(d['boxes']):
                color = HIT_COLOR if j in matched_pred else FALSE_COLOR
                ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec=color, lw=2))
                ax.text(box[0], box[1] - 3, '{} {:.2f}'.format(short[d['cls'][j]], d['conf'][j]),
                        color='white', fontsize=8, bbox=dict(fc=color, ec='none', pad=1))
            ax.set_title('{}: {}'.format(title, os.path.basename(d['path']).split('.rf.')[0]),
                         fontsize=9, loc='left')
    legend = [Line2D([], [], color=HIT_COLOR, lw=2, label='correct detection'),
              Line2D([], [], color=FALSE_COLOR, lw=2, label='false alarm'),
              Line2D([], [], color=MISS_COLOR, lw=2, ls='--', label='missed pot'),
              Line2D([], [], color='#9a9993', lw=1, ls='--', label='true pot (found)')]
    fig.legend(handles=legend, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('{}: example test-set detections'.format(name), x=0.01, ha='left', fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', nargs='+', required=True, help='One or more trained best.pt files (max 3).')
    ap.add_argument('--names', nargs='+', default=None,
                    help='A short label per --weights, e.g. yolo12s yolo26s. Default: the run folder name.')
    ap.add_argument('--data', required=True, help='Path to data.yaml')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                    help='Which images to draw and score. "test" for anything you report.')
    ap.add_argument('--out', default='figures', help='Folder to write the PNGs and summary.csv into.')
    ap.add_argument('--pr-conf', type=float, default=0.25,
                    help='Confidence cutoff for the confusion matrix and example images - same meaning '
                         'and default as evaluate.py\'s --pr-conf.')
    ap.add_argument('--iou', type=float, default=0.5, help='Box overlap needed to count as a hit (0-1).')
    ap.add_argument('--examples', type=int, default=4, help='Images per row in the examples figure.')
    ap.add_argument('--batch', type=int, default=16, help='Images per inference batch (a speed setting only).')
    args = ap.parse_args()

    if len(args.weights) > len(MODEL_COLORS):
        ap.error('at most {} models per figure (more gets unreadable - run it twice)'.format(len(MODEL_COLORS)))
    names = args.names or [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(w)))))
                           for w in args.weights]
    if len(names) != len(args.weights):
        ap.error('--names needs exactly one label per --weights file')

    import matplotlib
    matplotlib.use('Agg')  # no screen on a cluster node - write files only
    os.makedirs(args.out, exist_ok=True)

    class_names, items = load_split(args.data, args.split)
    print('{} split: {} images, {} true boxes'.format(args.split, len(items), sum(len(c) for _, c, _ in items)))

    models = list(zip(names, args.weights, MODEL_COLORS))
    scored = []
    summary = []
    for name, weights, color in models:
        print('Running {} ...'.format(name))
        # One pass at a very low cutoff; the higher cutoffs are applied afterwards
        # by filtering, so the model only has to look at each image once.
        preds = predict_all(weights, items, conf=0.001, batch=args.batch)
        s = score_model(items, preds, len(class_names), args.iou, args.pr_conf)
        scored.append((name, color, s))
        plot_confusion(name, s, class_names, args.pr_conf, os.path.join(args.out, '{}_confusion.png'.format(name)))
        plot_examples(name, s, class_names, args.examples, os.path.join(args.out, '{}_examples.png'.format(name)))

        at = min(s['sweep'], key=lambda r: abs(r[0] - args.pr_conf))
        best = s['sweep'][int(np.argmax(s['sweep'][:, 3]))]
        c = s['confusion']
        summary.append(dict(model=name, split=args.split, true_boxes=s['n_gt'],
                            conf=round(float(at[0]), 2), precision=round(float(at[1]), 3),
                            recall=round(float(at[2]), 3), f1=round(float(at[3]), 3),
                            best_f1_conf=round(float(best[0]), 2), best_f1=round(float(best[3]), 3),
                            found=int(c[:-1, :-1].sum()), missed=int(c[:-1, -1].sum()),
                            false_alarms=int(c[-1, :-1].sum())))

    plot_pr_and_sweep(scored, args.pr_conf, args.out)
    plot_training_curves(models, os.path.join(args.out, 'training_curves.png'))

    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print()
    for row in summary:
        print('{model}: P={precision} R={recall} F1={f1} at conf {conf} | best F1 {best_f1} at conf '
              '{best_f1_conf} | found {found}, missed {missed}, false alarms {false_alarms}'.format(**row))
    print('\nFigures written to {}/'.format(os.path.abspath(args.out)))


if __name__ == '__main__':
    main()

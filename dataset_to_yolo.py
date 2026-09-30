'''
Step 1 of 3: convert the raw dataset into the format Ultralytics YOLO expects.

WHY THIS SCRIPT EXISTS
The dataset (PINGEcosystem/sss-crab-pot-detection-ds on Hugging Face) stores each
image's crab-pot locations as a bounding box [x, y, w, h] in raw pixels, listed in
a metadata.jsonl file alongside the images — one JSON line per image. This is a
common, general-purpose format ("COCO-style"), but it's not what Ultralytics'
training code reads. YOLO expects one small .txt file per image, with each box
written as normalized (0-1, not pixels) CENTER coordinates rather than a corner:

    <class_id> <center_x> <center_y> <width> <height>

all divided by the image's width/height so they don't depend on image resolution.
This script does that conversion and also produces the small data.yaml file that
tells Ultralytics where the images live and what the class names are.

A NOTE ON TRUST: [x, y, w, h] could plausibly mean "top-left corner" or "center" —
formats disagree on this and get it wrong silently. Before writing this converter,
that ambiguity was resolved by drawing a box both ways on a real image and looking
at it: top-left-corner was the one that actually landed on a crab pot. That's the
convention used below.

USAGE
    python dataset_to_yolo.py --src <downloaded-dataset> --dst <output-folder>

    # To match the pre-trained packaged models' preprocessing (they only ever
    # predict one class, "Crab-Pot", having dropped the ambiguous label):
    python dataset_to_yolo.py --src ... --dst ... --drop-maybe
'''

import argparse
import json
import os
import shutil

from PIL import Image

# The dataset's Hugging Face split names (left) vs. the names Ultralytics expects
# (right) aren't identical - HF calls the middle split "valid", Ultralytics wants
# "val". This just renames the folder on the way out.
SPLIT_MAP = {'train': 'train', 'valid': 'val', 'test': 'test'}

# The two labels annotators used: a confident "Crab-Pot" and an ambiguous
# "Maybe-Crab-Pot" for targets that looked pot-like but weren't certain. Kept as
# two separate classes by default so the model can learn the difference; pass
# --drop-maybe to instead throw the ambiguous ones away entirely (matching how
# the paper's packaged models were prepared).
DEFAULT_CLASSES = ['Crab-Pot', 'Maybe-Crab-Pot']


def convert_split(src_split_dir, dst_images_dir, dst_labels_dir, class_to_id, symlink, drop_maybe):
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    metadata_path = os.path.join(src_split_dir, 'metadata.jsonl')
    n_images = 0
    n_boxes = 0
    n_dropped = 0

    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            file_name = rec['file_name']
            src_img_path = os.path.join(src_split_dir, file_name)
            if not os.path.exists(src_img_path):
                print('  WARNING: missing image, skipping: {}'.format(src_img_path))
                continue

            # Need the image's actual pixel size to convert the box below from
            # absolute pixels into the 0-1 fraction-of-image-size YOLO wants.
            with Image.open(src_img_path) as im:
                img_w, img_h = im.size

            dst_img_path = os.path.join(dst_images_dir, file_name)
            if symlink:
                # --symlink: point at the original file instead of copying it.
                # Instant and uses no extra disk, but only works when --dst is on
                # the same drive/filesystem as --src (a symlink can't cross onto
                # different physical storage the way a copy can).
                if os.path.lexists(dst_img_path):
                    os.remove(dst_img_path)
                os.symlink(os.path.abspath(src_img_path), dst_img_path)
            else:
                shutil.copyfile(src_img_path, dst_img_path)

            objects = rec.get('objects', {}) or {}
            bboxes = objects.get('bbox', []) or []
            categories = objects.get('category', []) or []

            lines = []
            for (x, y, w, h), category in zip(bboxes, categories):
                if category == 'Maybe-Crab-Pot' and drop_maybe:
                    n_dropped += 1
                    continue
                if category not in class_to_id:
                    print('  WARNING: unknown category "{}" in {}, skipping box'.format(category, file_name))
                    continue
                class_id = class_to_id[category]
                # The actual format conversion: [top-left x, top-left y, width,
                # height] in pixels -> [center x, center y, width, height] as a
                # fraction (0-1) of the image's own dimensions.
                cx = (x + w / 2.0) / img_w
                cy = (y + h / 2.0) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(class_id, cx, cy, nw, nh))
                n_boxes += 1

            # One .txt file per image, same base filename. An image with no
            # crab pots in it gets an EMPTY .txt file, not a missing one - that's
            # intentional. YOLO treats it as a confirmed "nothing here" example,
            # which helps the model learn what background/clutter looks like
            # instead of only ever seeing images that contain a pot.
            label_name = os.path.splitext(file_name)[0] + '.txt'
            with open(os.path.join(dst_labels_dir, label_name), 'w') as lf:
                lf.write('\n'.join(lines))
                if lines:
                    lf.write('\n')

            n_images += 1

    return n_images, n_boxes, n_dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='Path to the downloaded dataset folder (contains train/, valid/, test/)')
    ap.add_argument('--dst', required=True, help='Where to write the converted, YOLO-ready copy')
    ap.add_argument('--drop-maybe', action='store_true',
                     help='Throw away "Maybe-Crab-Pot" boxes and train on a single class ("Crab-Pot" '
                          'only). This is what the paper\'s packaged models did. Default: keep both '
                          'classes, so the model has to actually distinguish confident vs. ambiguous targets.')
    ap.add_argument('--symlink', action='store_true',
                     help='Link to the original images instead of copying them (fast, saves disk - only '
                          'works if --dst is on the same drive as --src). Leave this off unless you '
                          'specifically need the disk space back; plain copies are more portable.')
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    classes = ['Crab-Pot'] if args.drop_maybe else list(DEFAULT_CLASSES)
    class_to_id = {name: i for i, name in enumerate(classes)}

    print('Classes: {}'.format(classes))

    totals = {}
    for src_split, dst_split in SPLIT_MAP.items():
        src_split_dir = os.path.join(src, src_split)
        if not os.path.isdir(src_split_dir):
            print('Skipping missing split: {}'.format(src_split))
            continue
        print('Converting {} -> {}'.format(src_split, dst_split))
        n_images, n_boxes, n_dropped = convert_split(
            src_split_dir,
            os.path.join(dst, 'images', dst_split),
            os.path.join(dst, 'labels', dst_split),
            class_to_id,
            args.symlink,
            args.drop_maybe,
        )
        totals[dst_split] = (n_images, n_boxes, n_dropped)
        print('  {} images, {} boxes kept, {} Maybe-Crab-Pot boxes dropped'.format(n_images, n_boxes, n_dropped))

    # data.yaml is the one file Ultralytics actually reads to find everything:
    # where the images/labels are, and what each class ID (0, 1, ...) is called.
    yaml_lines = [
        'path: {}'.format(dst),
        'train: images/train',
        'val: images/val',
    ]
    if 'test' in totals:
        yaml_lines.append('test: images/test')
    yaml_lines.append('names:')
    for i, name in enumerate(classes):
        yaml_lines.append('  {}: {}'.format(i, name))

    data_yaml_path = os.path.join(dst, 'data.yaml')
    os.makedirs(dst, exist_ok=True)
    with open(data_yaml_path, 'w') as f:
        f.write('\n'.join(yaml_lines) + '\n')

    print('\nWrote {}'.format(data_yaml_path))
    print('Totals:', totals)


if __name__ == '__main__':
    main()

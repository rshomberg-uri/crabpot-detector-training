#!/bin/bash
# Make visualize.py's figures as a Slurm batch job (a few minutes on a GPU).
# Run from the repo folder, after `source cluster.env`:
#
#   sbatch slurm/visualize.sbatch figures/yolo12s runs/yolo12s/train/weights/best.pt
#   sbatch slurm/visualize.sbatch figures/compare runs/yolo12s/train/weights/best.pt runs/yolo26s/train/weights/best.pt
#
# First argument is the output folder; the rest are weights files.
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -o visualize_%j.log

OUT=$1
shift
source /etc/profile   # makes `module` available inside batch jobs on most clusters
source "$SLURM_SUBMIT_DIR/cluster.env" || exit 1
eval "$ENV_SETUP" || exit 1

python visualize.py --weights "$@" --data "$DATA" --out "$OUT"

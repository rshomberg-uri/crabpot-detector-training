#!/bin/bash
# Train one model as a Slurm batch job. Run from the repo folder:
#
#   source cluster.env              # once per login - see cluster.env.example
#   sbatch slurm/train.sbatch yolo12s     # or yolo26s
#
# Watch it with `squeue --me` and `tail -f train_<jobid>.log`.
# Output: runs/<arch>/train/weights/best.pt (plus the curves Ultralytics saves alongside).
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -o train_%j.log

ARCH=${1:-yolo12s}
source /etc/profile   # makes `module` available inside batch jobs on most clusters
source "$SLURM_SUBMIT_DIR/cluster.env" || exit 1
eval "$ENV_SETUP" || exit 1

python train.py --arch "$ARCH" --data "$DATA" --out "runs/$ARCH"

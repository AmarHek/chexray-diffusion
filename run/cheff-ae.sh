#!/bin/bash

#SBATCH --job-name=cheff-ae                   # Job name
#SBATCH --output=slurm_logs/output/cheff-ae_%j.out      # Standard output and error log
#SBATCH --error=slurm_logs/error/cheff-ae_%j.err        # Standard error
#SBATCH --partition=bl                        # Partition (queue) name
#SBATCH --gres=gpu:2                          # Request one GPU

python src/scripts/train_ldm.py -b configs/cheff-ae.yml -t --no-test --logdir=/home/hekalo_a/experiments/chexray-diffusion/
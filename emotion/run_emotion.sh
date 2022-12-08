#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=32GB

python run_emotion.py
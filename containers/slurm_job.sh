#!/bin/bash
#SBATCH --job-name=Mustard
#SBATCH --partition=long
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=result_%j.out
#SBATCH --error=result_%j.err

apptainer exec --nv ../container.sif ../build/lu_mustard -n=600 -t=2 --tiled --verify --verbose
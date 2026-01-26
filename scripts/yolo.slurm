#!/bin/bash

#SBATCH --ntasks        1
#SBATCH --partition     allgroups
#SBATCH --time          12:00:00
#SBATCH --mem           50G
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:rtx:2
#SBATCH --job-name      aperol
#SBATCH --output        output_%j.txt
#SBATCH --error         errors_%j.txt
#SBATCH --mail-user     q
#SBATCH --mail-type     ALL

cd train

srun singularity exec --nv ../../LFN/LFN_gpu.sif env PYTHONPATH=$PYTHONPATH:$(pwd)/train python yolo_training.py

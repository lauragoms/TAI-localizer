#!/bin/bash
#SBATCH --job-name=adaptive_disorder
#SBATCH --ntasks=10
#SBATCH --time=5:00:00
#SBATCH --partition=short
#SBATCH --output=adaptive_%j.out
#SBATCH --error=adaptive_%j.err

module purge

source /home/lgomez/miniconda3/etc/profile.d/conda.sh
conda activate peru_env

srun python adaptive_fig3D_cluster.py
#!/bin/bash
#SBATCH --job-name=adaptive_disorder
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --partition=short
#SBATCH --output=adaptive_%j.out
#SBATCH --error=adaptive_%j.err

module purge
module load MPI/GCC/8.3.1/openmpi-4.2.1
source /home/lgomez/miniconda3/etc/profile.d/conda.sh
conda activate peru_env

srun --mpi=pmi2 python adaptive_fig3D_cluster.py
#!/bin/bash
#SBATCH -J z2sig3D_R
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH -t 1-00:00:00
#SBATCH --array=0-49
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr

#SBATCH -o %x.%A.%a.out
#SBATCH -e %x.%A.%a.err

module purge
module load Miniforge3
module load mpi4py/3.1.4-gompi-2022b
source $(conda info --base)/etc/profile.d/conda.sh
conda activate peru_env

# 3D Z2 vs sigma, L=10, repulsion kappa_shift=3 r0, W=0, MJ=2, 100 reals.
# The SLURM array (0-49) parallelizes sigma = linspace(0,0.25,50)[SLURM_ARRAY_TASK_ID] (in r0).
# Cutoff scan, one array per R (R=1.3 is included as the reference on the same sigma grid):
#   for R in 1.05 1.1 1.15 1.2 1.25 1.3; do sbatch submit.sh --R $R; done
~/.conda/envs/peru_env/bin/python z2_vs_sigma_3D_cluster.py \
    --sigma-index "$SLURM_ARRAY_TASK_ID" "$@"

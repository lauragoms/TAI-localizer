#!/bin/bash
#SBATCH -J z2sig_R
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH -t 1-00:00:00
#SBATCH --array=0-49
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr

#SBATCH -o %x.%A.%a.out
#SBATCH -e %x.%A.%a.err

module purge
module load Miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate peru_env

# Z2 vs sigma, L=20, repulsion kappa_shift=3 r0, W=0, Delta=2, 100 reals.
# The SLURM array (0-49) parallelizes sigma = linspace(0,0.5,50)[SLURM_ARRAY_TASK_ID] (in r0).
# Cutoff scan, one array per R (R=1.3 already exists in appendix/finite_size_scaling/results_2d_v_3r0):
#   for R in 1.05 1.1 1.15 1.2 1.25; do sbatch submit.sh --R $R; done
~/.conda/envs/peru_env/bin/python z2_vs_sigma_2D_cluster.py \
    --sigma-index "$SLURM_ARRAY_TASK_ID" "$@"

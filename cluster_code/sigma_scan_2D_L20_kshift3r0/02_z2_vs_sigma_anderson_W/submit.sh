#!/bin/bash
#SBATCH -J z2sig_W
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

# Z2 vs sigma, L=20, repulsion kappa_shift=3 r0, R=1.3 r0, Delta=2, 100 reals.
# The SLURM array (0-49) parallelizes sigma = linspace(0,0.5,50)[SLURM_ARRAY_TASK_ID] (in r0).
# Anderson scan, one array per W (W=0 already exists in appendix/finite_size_scaling/results_2d_v_3r0):
#   for W in 2 4 6 8 10 12; do sbatch submit.sh --W $W; done
~/.conda/envs/peru_env/bin/python z2_vs_sigma_2D_cluster.py \
    --sigma-index "$SLURM_ARRAY_TASK_ID" "$@"

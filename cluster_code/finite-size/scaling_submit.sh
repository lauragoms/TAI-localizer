#!/bin/bash
#SBATCH -J locgap
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH -t 1-00:00:00
#SBATCH --array=0-49
##SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr
#SBATCH -o output/%A.%a.out
#SBATCH -e output/%A.%a.err

mkdir -p output
module purge
module load Miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nompi


ARG_ARRAY=("$@")
ID=${ARG_ARRAY[$SLURM_ARRAY_TASK_ID]}

~/.conda/envs/nompi/bin/python job_finite_size_scaling_3D.py "$ID"

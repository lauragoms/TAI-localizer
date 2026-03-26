#!/bin/bash
#SBATCH -J kappa
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --array=0-49
##SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr
#SBATCH --output=%A.%a.out
#SBATCH --error=%A.%a.err


source /home/lgomez/miniconda3/etc/profile.d/conda.sh
conda activate peru_env

ARG_ARRAY=("$@")
ID=${ARG_ARRAY[$SLURM_ARRAY_TASK_ID]}

/home/lgomez/miniconda3/envs/peru_env/bin/python job_for_3D_bc.py "$ID"
#!/bin/bash
#SBATCH -J localizer
#SBATCH -p general
#SBATCH --qos=regular
#SBATCH -t 1-00:00:00
#SBATCH --array=0-49
#SBATCH --mem=4G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr
#SBATCH --output=output/%A.%a.out
#SBATCH --error=output/%A.%a.err


module purge
module load Miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate peru_env


ARG_ARRAY=("$@")
ID=${ARG_ARRAY[$SLURM_ARRAY_TASK_ID]}

~/.conda/envs/peru_env/bin/python job_for_3D_a.py "$ID"
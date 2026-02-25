#!/bin/bash
#SBATCH --job-name=adaptive_disorder
#SBATCH --ntasks=100
#SBATCH --time=5:00:00
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --output=adaptive_%j.out
#SBATCH --error=adaptive_%j.err

module purge
module load Miniforge3  
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/grushin/.conda/envs/peru_env

srun -n $SLURM_NTASKS /home/grushin/.conda/envs/peru_env/bin/python -m mpi4py.futures adaptive_fig3D_cluster.py
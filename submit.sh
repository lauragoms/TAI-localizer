#!/bin/bash
#SBATCH --job-name=adaptive_disorder
#SBATCH --ntasks=100
#SBATCH --time=5:00:00
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --output=adaptive_%j.out
#SBATCH --error=adaptive_%j.err

module purge
module load gompi/2022b
module load mpi4py/3.1.4-gompi-2022b
module load Miniforge3   # si existe como módulo separado

source ~/peru_env/bin/activate

srun -n $SLURM_NTASKS python -m mpi4py.futures adaptive_fig3D_cluster.py
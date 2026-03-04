#!/bin/bash
#SBATCH --job-name=adaptive_disorder
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
##SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --output=adaptive_%j.out
#SBATCH --error=adaptive_%j.err
#SBATCH --mail-type=BEGIN,END                                                   
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr    

module purge
module load OpenMPI/4.1.4-GCC-11.3.0
module load Miniforge3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate peru_env

export OMPI_MCA_btl=tcp,self
export MPI4PY_MAX_WORKERS=$SLURM_NTASKS
mpiexec -np $SLURM_NTASKS ~/.conda/envs/peru_env/bin/python -m mpi4py.futures func_for_fig_4b_cluster.py
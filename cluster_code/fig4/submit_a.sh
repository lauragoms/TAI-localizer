#!/bin/bash
#SBATCH --job-name=bhz_a
#SBATCH --ntasks=49
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
##SBATCH --mem=100G
#SBATCH --time=24:00:00
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
mpiexec -np $SLURM_NTASKS ~/.conda/envs/peru_env/bin/python -m mpi4py.futures job_4a_cluster.py
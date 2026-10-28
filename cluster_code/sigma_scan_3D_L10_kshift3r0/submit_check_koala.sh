#!/bin/bash
#SBATCH -J check_koala
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH -t 00:10:00
#SBATCH --mem=2G

#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

module purge
module load Miniforge3
module load mpi4py/3.1.4-gompi-2022b
source $(conda info --base)/etc/profile.d/conda.sh
conda activate peru_env

# Prints which koala peru_env uses; compare with the local reference in README.md.
#   sbatch submit_check_koala.sh   ->   cat check_koala.<jobid>.out
~/.conda/envs/peru_env/bin/python check_koala.py

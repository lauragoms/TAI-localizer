#!/bin/bash                                                                  
#SBATCH -J tai_loc_job                                                    
#SBATCH -p general
                                                                                                            
##SBATCH -t 3-00:00:00   
                                                        
                                                                       
#SBATCH --mail-type=BEGIN,END                                                   
#SBATCH --mail-user=laura.gomez-paz@neel.cnrs.fr     

#SBATCH --qos=regular

##SBATCH --array=0-50           # 30 tareas
#SBATCH --cpus-per-task=1
##SBATCH --mem=2G



mkdir -p output
#SBATCH -o=/output/%N.%j.%a.out
#SBATCH -e=/output/%N.%j.%a.err


# Print job information
echo "==================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Argument received: $@ "
echo "Running on cluster: $SLURM_CLUSTER_NAME"
echo "Running on host: $SLURM_NODELIST"
echo "Job name: $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMITDIR"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "==================================="

# you may not place bash commands before the last SBATCH directive              

module purge

module load Miniforge3

source $(conda info --base)/etc/profile.d/conda.sh

conda activate peru_env

##ARG_ARRAY=("$@")

# Select value corresponding to array index
##ID=${ARG_ARRAY[$SLURM_ARRAY_TASK_ID]}

# =============================
#  Run Python job
# =============================
mkdir -p output/seed_$ID

~/.conda/envs/peru_env/bin/python func_for_fig_kappa3Da_cluster.py "$ID" > output/seed_$ID/output.log 2> output/seed_$ID/error.log

wait  


# End of job
echo "==================================="
echo "Job finished at $(date)"
echo "==================================="

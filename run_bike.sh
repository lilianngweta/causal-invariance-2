#!/bin/bash
#SBATCH --job-name=bike_hsic
#SBATCH --output=logs_bike/hsic_%A_%a.out
#SBATCH --array=0-959
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH -D /gpfs/u/home/RLML/RLMLngwt/scratch/causal-invariance-2
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngwetl@rpi.edu

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source /gpfs/u/home/RLML/RLMLngwt/scratch/miniconda3/etc/profile.d/conda.sh
conda activate invariance_env

srun python khsic_approach_bike_sharing_data_no_coefficients.py $SLURM_ARRAY_TASK_ID



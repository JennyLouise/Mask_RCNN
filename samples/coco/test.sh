#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jw22g14@soton.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
python3 test.py $SLURM_ARRAY_TASK_ID

#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jw22g14@soton.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00   
python3 job_array_nnet.py $SLURM_ARRAY_TASK_ID '/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/labels/coco/'

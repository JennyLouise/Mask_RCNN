#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jw22g14@soton.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00   
python3 job_array_nnet.py $SLURM_ARRAY_TASK_ID '/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/images/processed/image/i20180805_215810/histogram_normalised/no_distortion_correction/not_rescaled/'
# '/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/labels/coco/large_megafauna_acfr_pipeline/' 
# '/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/images/processed/image/i20180805_215810/debayered/colour_balance/colour_balanced_point1/'
# '/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/images/processed/image/i20180805_215810/greyworld_correction/distortion_correction/not_rescaled/'

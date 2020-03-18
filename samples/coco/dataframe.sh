#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jw22g14@soton.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00  
#SBATCH --partition=scavenger 
python3 test.py

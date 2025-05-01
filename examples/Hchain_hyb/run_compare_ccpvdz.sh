#!/bin/bash
#SBATCH -o log-%j
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -A DMR21001
#SBATCH -p development
#SBATCH -t 2:00:00

# set python path
PYTHON="/work2/09730/whe1/anaconda3/envs/ML_DFT/bin/python3"
# Get the list of nodes

$PYTHON compare_hyb.py --start 0 --end 22 --atoms H6

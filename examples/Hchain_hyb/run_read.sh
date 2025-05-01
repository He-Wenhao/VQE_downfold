#!/bin/bash
#SBATCH -o log-%j
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -A DMR21001
#SBATCH -p development
#SBATCH -t 2:00:00

# set python path
PYTHON="/work2/09730/whe1/anaconda3/envs/ML_DFT/bin/python3"
# Get the list of nodes
NODELIST=$(scontrol show hostname $SLURM_NODELIST)
NODES=($NODELIST)

# Number of nodes
NUM_NODES=${#NODES[@]}

# Task parameters
TASKS_PER_NODE=92
ATOMS="H6"

# Loop through the nodes and assign tasks
for i in $(seq 0 $((NUM_NODES-1))); do
    START=$((i * TASKS_PER_NODE))
    END=$(((i + 1) * TASKS_PER_NODE - 1))
    NODE=${NODES[i]}

    # Run the task on the current node
    srun --nodes=1 --nodelist=$NODE $PYTHON read.py --atoms $ATOMS --start $START --end $END &
done

# Wait for all background jobs to finish
wait
echo done

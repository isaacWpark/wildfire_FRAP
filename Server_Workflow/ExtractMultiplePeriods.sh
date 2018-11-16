#!/bin/bash

# set output and error output filenames, %j will be replaced by Slurm with the jobid
#SBATCH -o testing%j.out
#SBATCH -e testing%j.err 
 
# single node in the "short" partition
#SBATCH -N 1
#SBATCH -p 128gb

# half hour timelimit
#SBATCH -t 48:00:00


#SBATCH --mail-type=ALL 
#SBATCH --mail-user=mmann1123@gwu.edu  


source activate /groups/manngroup/CondaEnvP3.6/

python ~/wildfire_FRAP/Server_Workflow/ExtractMultiplePeriods.py

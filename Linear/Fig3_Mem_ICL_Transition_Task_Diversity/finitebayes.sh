#!/bin/bash
# longerrun_d80_alpha10.sbatch
# 
#SBATCH --job-name=longerrun_d80_alpha10
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/appendix/outputfiledump/longerrun_d80_alpha10_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/appendix/outputfiledump/longerrun_d80_alpha10_%a.err
#SBATCH --array=1-50
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="appendix"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py 80 $newdir 10 $SLURM_ARRAY_TASK_ID 0.1
#!/bin/bash
# d80_alpha40_500iter.sbatch
# 
#SBATCH --job-name=d80_alpha40_500iter
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/d80_alpha40_500iter_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/d80_alpha40_500iter_%a.err
#SBATCH --array=1-40
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="pnas"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py 100 $newdir 40 $SLURM_ARRAY_TASK_ID 0.1
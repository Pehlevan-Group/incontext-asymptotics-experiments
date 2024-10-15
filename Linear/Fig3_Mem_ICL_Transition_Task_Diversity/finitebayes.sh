#!/bin/bash
# alpha100redo.sbatch
# 
#SBATCH --job-name=alpha100redo
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/pnas/outputfiledump/alpha100redo_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/pnas/outputfiledump/alpha100redo_%A.err
#SBATCH --array=23-40
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="pnas"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py $newdir 100 $SLURM_ARRAY_TASK_ID
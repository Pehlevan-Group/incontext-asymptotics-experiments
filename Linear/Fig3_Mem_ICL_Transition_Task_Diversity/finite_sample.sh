#!/bin/bash
# finitesample_a1_k10.sbatch
# 
#SBATCH --job-name=finitesample_a1_k10
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/pnas/outputfiledump/finitesample_a1_k10_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/pnas/outputfiledump/finitesample_a1_k10_%a.err
#SBATCH --array=10-14
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="pnas"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finite_sample_run.py $SLURM_ARRAY_TASK_ID $newdir 1 10 0.1 100000
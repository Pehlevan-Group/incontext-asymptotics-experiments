#!/bin/bash
# largekappa_alpha1_d50.sbatch
# 
#SBATCH --job-name=largekappa_alpha1_d50
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/pnas/outputfiledump/largekappa_alpha1_d50_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/pnas/outputfiledump/largekappa_alpha1_d50_%a.err
#SBATCH --array=1-9
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="pnas"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py 50 $newdir 1 $SLURM_ARRAY_TASK_ID 0.1
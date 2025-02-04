#!/bin/bash
# PNAS_final_FIG4_a40_t8_d140.sbatch
# 
#SBATCH --job-name=PNAS_final_FIG4_a40_t8_d140
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/kapparuns/outputs/PNAS_final_FIG4_a40_t8_d140_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/kapparuns/outputs/PNAS_final_FIG4_a40_t8_d140_%a.err
#SBATCH --array=1-5
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
parentdir="results"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python kappa.py 140 40 8 $SLURM_ARRAY_TASK_ID $newdir
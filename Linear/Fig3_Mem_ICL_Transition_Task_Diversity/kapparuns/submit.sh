#!/bin/bash
# a100_t4_d70_kempner.sbatch
# 
#SBATCH --job-name=a100_t4_d70_kempner
#SBATCH --gpus 1
#SBATCH -t 1-00:00:00
#SBATCH -p kempner
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/kapparuns/results/a100_t4_d70_kempner_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/kapparuns/outputs/a100_t4_d70_kempner_%a.err
#SBATCH --array=1-5
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source activate try4
python kappa.py 70 100 4 $SLURM_ARRAY_TASK_ID
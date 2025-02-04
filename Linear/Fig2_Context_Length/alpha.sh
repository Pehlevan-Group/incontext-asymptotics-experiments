#!/bin/bash
# FIG2B_E.sbatch
# 
#SBATCH --job-name=FIG2B_E
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig2_Context_Length/dump/FIG2B_E_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/incontext-asymptotics-experiments/Linear/Fig2_Context_Length/dump/FIG2B_E_%a.err
#SBATCH --array=1-4
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
parentdir="PNAS_VERIFY"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python fixed_kappa_sim_alpha.py $newdir 1.5 $SLURM_ARRAY_TASK_ID
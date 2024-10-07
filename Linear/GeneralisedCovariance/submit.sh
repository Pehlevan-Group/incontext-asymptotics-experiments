#!/bin/bash
# largerun_multiply2.sbatch
# 
#SBATCH --job-name=largerun_multiply2
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Alpha_Shift/outputs/largerun_multiply2%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Alpha_Shift/outputs/largerun_multiply2%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python generalcovariance.py 80 1 1
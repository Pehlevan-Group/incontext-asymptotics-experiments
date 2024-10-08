#!/bin/bash
# 0p3_noshift_kappa0p25.sbatch
# 
#SBATCH --job-name=0p3_noshift_kappa0p25
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Alpha_Shift/outputs/0p3_noshift_kappa0p25_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Alpha_Shift/outputs/0p3_noshift_kappa0p25_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python contextshift.py 100 0.3 0.25 2 0.3 0.3
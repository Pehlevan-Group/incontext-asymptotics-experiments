#!/bin/bash
# d20alpha1_1000gradstep.sbatch
# 
#SBATCH --job-name=d20alpha1_1000gradstep
#SBATCH -c 5
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Nonlinear/experiment/explicitlylinearattention/outputs/d20alpha1_1000gradstep_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Nonlinear/experiment/explicitlylinearattention/outputs/d20alpha1_1000gradstep_%a.err
#SBATCH --array=1-210%50
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

# module load python/3.10.12-fasrc01
# module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate torchenv
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    tauind=$(( ($1 - 1) / 10 ))
    avgind=$(( ($1 - 1) % 10 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

newdir="runs/${SLURM_JOB_NAME}"
mkdir "$newdir"
python redo.py $newdir 20 $tauind $avgind
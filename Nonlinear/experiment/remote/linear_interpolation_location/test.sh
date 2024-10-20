#!/bin/bash
# 80lowtrain.sbatch
# 
#SBATCH --job-name=80lowtrain
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Nonlinear/experiment/remote/linear_interpolation_location/outputdump/80lowtrain_%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/incontext-asymptotics-experiments/Nonlinear/experiment/remote/linear_interpolation_location/outputdump/80lowtrain_%A_%a.err
#SBATCH --array=1-105%30
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    tauind=$(( ($1 - 1) / 5 ))
    avgind=$(( ($1 - 1) % 5 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="pnas"
newdir="$parentdir/${SLURM_JOB_NAME}"
pkldir="$parentdir/${SLURM_JOB_NAME}/pickles"
errdir="$parentdir/${SLURM_JOB_NAME}/errors"
mkdir "$newdir"
mkdir "$pkldir"
mkdir "$errdir"
python run.py $newdir 80 $tauind $avgind
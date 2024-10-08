#!/bin/bash
# a2p5_shift1_hype.sbatch
# 
#SBATCH --job-name=a2p5_shift1_hype
#SBATCH -c 10
#SBATCH -t 2-00:00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/alpha_shift/outputdump/a2p5_sha2p5_shift1_hypeift1%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/alpha_shift/outputdump/a2p5_shift1_hype%A_%a.err
#SBATCH --mail-type=END
#SBATCH --array=1-10
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

parentdir="resultsdump"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
pkldir="$parentdir/job_${SLURM_JOB_NAME}/pickles"
mkdir "$newdir"
mkdir "$pkldir"
python run.py $newdir 20 2.5 1 0.125 $SLURM_ARRAY_TASK_ID 0 0 
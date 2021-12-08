#!/bin/bash
# Job name:
#SBATCH --job-name=socialgame_train
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit (8hrs):
#SBATCH --time=08:00:00
#
# Run 8 examples concurrently
#SBATCH --array=0-2

SINGULARITY_CACHE_DIR=/global/scratch/users/lucas_spangher/transactive-control-social-game

if test -f austin_docker.sif; then
    echo "docker image exists"
    singularity exec --nv --workdir ./tmp --bind $(pwd):$HOME --bind "$LDIR:$HOME/.local" sh -c "python feudal_trainer.py"
else
    singularity build austin_docker.sif library://yanlarry/default/singularity_phnet:v1
fi
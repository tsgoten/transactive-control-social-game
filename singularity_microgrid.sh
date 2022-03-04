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
#Number of GPUs, this can be in the format of “gpu:[1-4]“, or “gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit (8hrs):
#SBATCH --time=08:00:00
#
# Run 8 examples concurrently
#SBATCH --array=0
BASE_DIR=/global/scratch/users/$USER
LDIR=$BASE_DIR/.local$SLURM_ARRAY_TASK_ID
LOGDIR_BASE=$BASE_DIR/logs
rm -rf $LDIR
mkdir -p $LDIR
rm -rf /global/home/users/lucas_spangher/.cache/pip
SINGULARITY_IMAGE_LOCATION=/global/scratch/users/$USER
SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
export SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
SINGULARITY_TEMPDIR=$BASE_DIR/tmp
export SINGULARITY_TEMPDIR=$BASE_DIR/tmp
SINGULARITY_CACHE_DIR=/global/scratch/users/lucas_spangher/transactive-control-social-game
export WANDB_API_KEY=7385069f57b00860da0e7add0bdc6eba19fb07cd
if test -f austin_docker2.sif; then
  echo “docker image exists”
else
  singularity build austin_docker.sif docker://lucasspangher/socialgame_v1:austin
fi
singularity run --nv --workdir ./tmp --bind $(pwd):$HOME --bind “$LDIR:$HOME/.local” austin_docker.sif sh -c ‘./singularity_preamble_new.sh && ./microgrid_experiment_script.sh’










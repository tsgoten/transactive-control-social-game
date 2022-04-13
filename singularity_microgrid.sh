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
SINGULARITY_CACHE_DIR=/global/scratch/users/djang/transactive-control-social-game
export WANDB_API_KEY=87928bf7ce62528545fe624701ab2f3aa25a7547
if test -f austin_docker.sif; then
  echo “docker image exists”
else
  singularity build austin_docker.sif docker://lucasspangher/socialgame_v1:austin
fi
singularity run --nv --workdir ./tmp --bind $(pwd):$HOME --bind "$LDIR:$HOME/.local" austin_docker.sif sh -c "./singularity_preamble_new.sh && wandb agent social-game-rl/energy-demand-response-game/$1 --count 1"

#singularity run --nv --workdir ./tmp --bind $(pwd):$HOME --bind "$LDIR:$HOME/.local" austin_docker.sif sh -c './singularity_preamble_new.sh && python ExperimentRunner.py -w --gym_env=microgrid_multi --custom_config=configs/mg_configs/simple_manyagents.json --num_gpus=1 --num_mg_workers=2 --num_steps=100000 --batch_size=16 --use_hnet --hnet_embedding_dim=322 --hnet_lr=0.004697687912444648 --hnet_num_hidden=198 --hnet_num_layers=2 --hnet_num_local_steps=20 --learning_rate=0.004223331881088467 --n_layers=1 --ppo_clip_param=0.6098559462062045 --ppo_num_sgd_iter=2 --sizes=8 --use_agg_data'









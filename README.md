# Transactive Control Social Game
Code meant to support and simulate the Social Game that will be launched in 2021. Elements of transactive control and behavioral engineering will be tested and designed here

## Installation
(1) pip install requirements.txt
(2) docker build .

## Usage
1. Run `./run.sh` from the root of the repo. This will put you in a shell in the docker container with the `rl_algos/logs` directory mounted
2. Run `python3 ExperimentRunner.py sac test_experiment` in the docker container to start an experiment with the name `test_experiment`
3. Run `tensorboard --logdir rl_algos/logs` from outside the docker container to view the logs
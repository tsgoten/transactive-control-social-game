# Transactive Control Social Game
Code meant to support and simulate the Social Game that will be launched in 2021. Elements of transactive control and behavioral engineering will be tested and designed here

## Installation
1. Clone the repo
2. Install [dvc](https://dvc.org/doc/install) (with google drive support)
  * On linux this is `pip install 'dvc[gdrive]'`
3. Install Docker, if you have not already. 
4. Run `python3 -m dvc remote add -d gdrive gdrive://1qaTn6IYd3cpiyJegDwwEhZ3LwrujK3_x`
5. Run `python3 -m dvc pull`
6. Run `pip install -r requirements.txt`


## Usage
1. Run `./run.sh` from the root of the repo. This will put you in a shell in the docker container with the `rl_algos/logs` directory mounted
2. Run `python3 ExperimentRunner.py sac test_experiment` in the docker container to start an experiment with the name `test_experiment`
3. Run `tensorboard --logdir rl_algos/logs` from outside the docker container to view the logs

### Issues
If you're having trouble running docker or the `ExperimentRunner.py` file. Please try running `python ExperimentRunner.py` and debug from there. 

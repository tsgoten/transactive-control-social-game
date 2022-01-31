import argparse
from gym import spaces

import numpy as np
import pandas as pd 
import ray
from ray import tune
import ray.rllib.agents.sac.sac as sac

import utils
import os

from gym_microgrid.envs.feudal_env import (FeudalSocialGameHourwise, FeudalMicrogridEnvHigherAggregator)
from gym_microgrid.envs.feudal_microgrid_baselines_env import (FeudalMicrogridEnvOnlyLowerBaselineEnv)
from gym_socialgame.envs.socialgame_env import SocialGameEnvRLLib
from custom_callbacks import (HierarchicalCallbacks, CustomCallbacks, HierarchicalMultigridCallbacks)
import pdb

import wandb

parser = argparse.ArgumentParser()

parser.add_argument(
    "--action_space_string",
    help="action space for algorithm",
    default="continuous",
)
parser.add_argument(
    "--policy_type",
    help="Type of Policy (e.g. MLP, LSTM) for algo",
    default="mlp",
    choices=["mlp", "lstm"],
)
parser.add_argument(
    "--reward_function",
    help="reward function to test",
    type=str,
    default="log_cost_regularized",
    choices=["scaled_cost_distance", "log_cost_regularized", "log_cost", "scd", "lcr", "lc", "market_solving", "profit_maximizing"],
)

# Environment Arguments

parser.add_argument(
    "--gym_env",
    help="feudal or flat",
    default="feudal_timewise",
    choices=[
        "feudal_timewise", "socialgame_env", 
        "feudal_spatial", "feudal_spatial_lower_baseline"
        ]
)
parser.add_argument(
    "--response_type_string",
    help="Player response function (l = linear, t = threshold_exponential, s = sinusoidal",
    type=str,
    default="l",
    choices=["l", "t", "s"],
)
# Experiment Arguments
parser.add_argument(
    "--exp_name",
    help="experiment_name",
    type=str,
    default="experiment"
)
parser.add_argument(
    "--wandb_group",
    help="Group to group wandb exps under",
    type=str,
    default="scratch_experiments"
)
parser.add_argument(
    "--num_steps",
    help="Number of timesteps to train algo",
    type=int,
    default=50000,
)
parser.add_argument(
    "--energy_in_state",
    help="Whether to include energy in state (default = F)",
    action="store_true"
)
parser.add_argument(
    "--price_in_state",
    help="Whether to include price in state (default = F)",
    action="store_false"
)
parser.add_argument(
    "--batch_size",
    help="Batch Size for sampling from replay buffer",
    type=int,
    default=5,
    choices=[i for i in range(1, 30)],
)
parser.add_argument(
    "--one_day",
    help="Specific Day of the year to Train on (default = 15, train on day 15)",
    type=int,
    default=15,
    choices=[i for i in range(365)],
)
parser.add_argument(
    "--manual_tou_magnitude",
    help="Magnitude of the TOU during hours 5,6,7. Sets price in normal hours to 0.103.",
    type=float,
    default=.4
)
parser.add_argument(
    "--pricing_type",
    help="time of use or real time pricing",
    type=str,
    choices=["TOU", "RTP"],
    default="TOU",
)
parser.add_argument(
    "--number_of_participants",
    help="Number of players ([1, 20]) in social game",
    type=int,
    default=10,
    choices=[i for i in range(1, 21)],
)
parser.add_argument(
    "--learning_rate",
    help="learning rate of the the agent",
    type=float,
    default=3e-4,
)
parser.add_argument(
    "--lower_level_reward_type",
    help="reward function for lower level agent",
    type=str,
    default="l2",
)
# Logging Arguments
parser.add_argument(
    "-w",
    "--wandb",
    help="Whether to upload results to wandb. must have wandb key.",
    action="store_true"
)
parser.add_argument(
    "--log_path",
    help="Base directory for tensorboard logs",
    type=str,
    default="./logs/"
)
parser.add_argument(
    "--bulk_log_interval",
    help="Interval at which to save bulk log information",
    type=int,
    default=50,
)
parser.add_argument(
    "--bin_observation_space",
    help= "Whether to bin the observations.",
    action="store_true"
)
parser.add_argument(
    "--smirl_weight",
    help="Whether to run with SMiRL. When using SMiRL you must specify a weight.",
    type = float,
    default=None,
)
parser.add_argument(
    "--circ_buffer_size",
    help="Size of circular smirl buffer to use. Will use an unlimited size buffer in None",
    type = float,
    default=None,
)
# Machine Arguments
parser.add_argument(
    "-l",
    "--local_mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true"
)
parser.add_argument(
    "--num_inner_steps",
    help="Number of local model training steps",
    type= int,
    default=1
)


if __name__== "__main__":
    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project="energy-demand-response-game", 
            entity="social-game-rl",
            group=args.wandb_group)
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)

    out_path = os.path.join(args.log_path, str(pd.datetime.today()) + "_bulk_data.h5")

    if args.gym_env == "feudal_timewise":

        upper_level_obs_space = spaces.Box(
            low = -np.inf, high = np.inf, 
            shape = (20,), dtype = np.float32)
        upper_level_action_space = spaces.Box(
            low = -1, high = 1, 
            shape = (5,), dtype = np.float32)
        lower_level_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(5,), dtype=np.float32)
        lower_level_action_space = spaces.Box(
            low = -1, high = 1, 
            shape = (2,), dtype = np.float32)

        policies = {"higher_level_agent": (
            None, upper_level_obs_space, 
            upper_level_action_space, {"gamma": 1})}
        for i in range(5):
            policies["lower_level_agent_{}".format(i)] = (
                None, lower_level_obs_space, 
                lower_level_action_space, {"gamma": 1})

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return agent_id

        config = {
            "env": FeudalSocialGameHourwise,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "env_config": vars(args)
        }

        agent_keys = [f"lower_level_agent_{i}" for i in range(5)]
        agent_keys += ["higher_level_agent"]


        callbacks = HierarchicalCallbacks(
            log_path=out_path, 
            save_interval=args.bulk_log_interval, 
            obs_dim=20, 
            agent_keys = agent_keys)

        config["callbacks"] = lambda: callbacks
        logger_creator = utils.custom_logger_creator(args.log_path)

        trainer = sac.SACTrainer(
            env=FeudalSocialGameHourwise, 
            config=config,
            logger_creator=logger_creator,
        )

    elif args.gym_env == "feudal_spatial":
        upper_level_obs_space = spaces.Box(
            low = -np.inf, high = np.inf, 
            shape = (24 * (2 + 6),), dtype = np.float64)
        upper_level_action_space = spaces.Box(
            low = -1, high = 1, 
            shape = (48,), dtype = np.float32)
        lower_level_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(24 * 6,), dtype=np.float64)
        lower_level_action_space = spaces.Box(
            low = -1, high = 1, 
            shape = (48,), dtype = np.float32)

        policies = {"higher_level_agent": (
            None, upper_level_obs_space, 
            upper_level_action_space, {"gamma": 1})}
        for i in range(6):
            policies["lower_level_agent_{}".format(i)] = (
                None, lower_level_obs_space, 
                lower_level_action_space, {"gamma": 1})

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return agent_id

        config = {
            "env": FeudalMicrogridEnvHigherAggregator,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "env_config": vars(args)
        }

        agent_keys = [f"lower_level_agent_{i}" for i in range(6)]
        agent_keys += ["higher_level_agent"]


        callbacks = HierarchicalMultigridCallbacks(
            log_path=out_path, 
            save_interval=args.bulk_log_interval, 
            obs_dim=20,  ## <<- this doesn't effect things 
            agent_keys = agent_keys)

        config["callbacks"] = lambda: callbacks
        logger_creator = utils.custom_logger_creator(args.log_path)

        trainer = sac.SACTrainer(
            env=FeudalMicrogridEnvHigherAggregator, 
            config=config,
            logger_creator=logger_creator,
        )

    elif args.gym_env == "feudal_spatial_lower_baseline":
        lower_level_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(24 * 6,), dtype=np.float64)
        lower_level_action_space = spaces.Box(
            low = -1, high = 1, 
            shape = (48,), dtype = np.float32)

        policies = {}
        for i in range(6):
            policies["lower_level_agent_{}".format(i)] = (
                None, lower_level_obs_space, 
                lower_level_action_space, {"gamma": 1})

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return agent_id

        config = {
            "env": FeudalMicrogridEnvOnlyLowerBaselineEnv,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "env_config": vars(args)
        }

        agent_keys = [f"lower_level_agent_{i}" for i in range(6)]

        callbacks = HierarchicalMultigridCallbacks(
            log_path=out_path, 
            save_interval=args.bulk_log_interval, 
            obs_dim=20,  ## <<- this doesn't effect things 
            agent_keys = agent_keys)

        config["callbacks"] = lambda: callbacks
        logger_creator = utils.custom_logger_creator(args.log_path)

        trainer = sac.SACTrainer(
            env=FeudalMicrogridEnvOnlyLowerBaselineEnv, 
            config=config,
            logger_creator=logger_creator,
        )

    elif args.gym_env=="socialgame_env":
        args.price_in_state = True
        args.energy_in_state = True
        config = {
            "env": SocialGameEnvRLLib,
            "env_config": vars(args)
        }
        callbacks = CustomCallbacks(
            log_path=out_path, 
            save_interval=args.bulk_log_interval, 
            obs_dim=20,
            )
        config["callbacks"] = lambda: callbacks
        logger_creator = utils.custom_logger_creator(args.log_path)

        trainer = sac.SACTrainer(
            env=SocialGameEnvRLLib, 
            config=config,
            logger_creator=logger_creator,)

    training_steps = 0 

    while training_steps < args.num_steps:
        print("in training loop")
        result = trainer.train()
        training_steps = result["timesteps_total"]
        log = result # {name: result[name] for name in to_log}
        print(f"------- training step {training_steps}-------")
        print(log)


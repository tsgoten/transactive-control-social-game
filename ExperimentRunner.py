import argparse
import gym
import numpy as np
import os


"""
Parse the experiment arguments and configurations.
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--library",
    help = "What RL Library backend is in use",
    type = str,
    default = "rllib",
    choices = ["rllib", "tune"]
)

# Algorithm Arguments
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    type=str,
    default="sac",
    choices=["sac", "ppo", "maml", "uc_bandit"]
)
parser.add_argument(
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
    help="Which Gym Environment you wihs to use",
    type=str,
    choices=["socialgame"],
    default="socialgame"
)
parser.add_argument(
    "--env_id",
    help="Environment ID for Gym Environment",
    type=str,
    choices=["v0", "monthly"],
    default="v0",
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
    "--num_steps",
    help="Number of timesteps to train algo",
    type=int,
    default=50000,
)
parser.add_argument(
    "--energy_in_state",
    help="Whether to include energy in state (default = F)",
    type=str,
    default="F",
    choices=["T", "F"],
)
parser.add_argument(
    "--price_in_state",
    help="Whether to include price in state (default = F)",
    type=str,
    default="T",
    choices=["T", "F"],
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

# Logging
parser.add_argument(
    "-w",
    "--wandb",
    help="Whether to upload results to wandb. must have wandb key.",
    action="store_true"
)
parser.add_argument(
    "--base_log_dir",
    help="Base directory for tensorboard logs",
    type=str,
    default="./logs/"
)
parser.add_argument(
    "--bulk_log_interval",
    help="Interval at which to save bulk log information",
    type=int,
    default=10000
)

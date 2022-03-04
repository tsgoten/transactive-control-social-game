import json
import argparse
import os
import random
def add_args(parser):
    parser.add_argument(
        "--num_envs",
        type=int,
        default = 3,
        help="Number of environments to generate configs for")
    parser.add_argument(
        "--custom_config",
        type=str,
        default = "configs/mg_.json",
        help="Path to custom config file"
    )
    parser.add_argument(
        "--number_of_participants",
        nargs='+',
        type=int,
        default = [10],
        help="List of number_of_participants; if len < num_envs, use first as default")
    parser.add_argument(
        "--complex_batt_pv_scenario",
        nargs='+',
        type=int,
        default = [1],
        help="List of complex_batt_pv_scenario; if len < num_envs, use first as default")
    parser.add_argument(
        "--num_mg_optim_steps",
        nargs='+',
        type=int,
        default = [10000],
        help="List of num_mg_optim_steps; if len < num_envs, use first as default")
    parser.add_argument(
        "--max_episode_steps",
        nargs='+',
        type=int,
        default = [365],
        help="List of max_episode_steps; if len < num_envs, use first as default")
    return parser
    
if __name__ == '__main__':
    configs = []
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    num_envs = args.num_envs
    number_of_participants = args.number_of_participants
    if len(number_of_participants) < num_envs:
        number_of_participants = [number_of_participants[0] for i in range(num_envs)]
    complex_batt_pv_scenario = args.complex_batt_pv_scenario
    if len(complex_batt_pv_scenario) < num_envs:
        complex_batt_pv_scenario = [complex_batt_pv_scenario[0] for i in range(num_envs)]
    num_mg_optim_steps = args.num_mg_optim_steps
    if len(num_mg_optim_steps) < num_envs:
        num_mg_optim_steps = [num_mg_optim_steps[0] for i in range(num_envs)]
    max_episode_steps = args.max_episode_steps
    if len(max_episode_steps) < num_envs:
        max_episode_steps = [max_episode_steps[0] for i in range(num_envs)]

    
    for i in range(num_envs):
        config = {
            # "action_space_string": "continuous",
            "number_of_participants": number_of_participants[i],
            # "one_day": 0,
            # "energy_in_state": False,
            # "reward_function": "market_solving",
            "complex_batt_pv_scenario": complex_batt_pv_scenario[i],
            # "exp_name": None,
            # "smirl_weight=": None,
            "num_mg_optim_steps": num_mg_optim_steps[i],
            # "num_mg_workers": 10,
            "max_episode_steps": max_episode_steps[i],
            # "starting_day": None, 
            # "no_rl_control": False,
            # "no_external_grid": False
        }
        configs.append(config)

    with open(args.custom_config, "w") as f:
        json.dump(configs, f, indent=4)


import json
import argparse
import os
import random

"""
parameters to test (within an individual microgrid):
    - number of buildings
    - how likely a building is to have pv/batt
    - mean and variance of the building pv/batt
    - 

between grids:
    - num buildings
    - proportion of buildings with pv/batt
    - mean and var of building pv/batt

axes:
    - differences in number
    - differences in total capacity
        - same total
    - differences in distribution
        - concentration, variance
        - mean

"""


def add_args(parser):
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default = 3,
        help="Number of scenarios to generate")
    parser.add_argument(
        "--custom_scenario",
        type=str,
        default = "configs/mg_.json",
        help="Path to custom config file")
    parser.add_argument(
        "--num_buildings",
        nargs='+',
        type=int,
        default = [10],
        help="List of num_buildings; if len < num_envs, use first as default")
    parser.add_argument(
        "--pv_mean",
        nargs='+',
        type=int,
        default = [0],
        help="List of pv means; if len < num_envs, use first as default")
    parser.add_argument(
        "--pv_var",
        nargs='+',
        type=int,
        default = [0],
        help="List of pv variances; if len < num_envs, use first as default")
    parser.add_argument(
        "--pv_bern",
        nargs='+',
        type=int,
        default = [0],
        help="List of pv Bernoulli probablities; if len < num_envs, use first as default")
    return parser
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    scenarios = []
    num_scenarios = args.num_scenarios

    number_of_participants = args.number_of_participants
    if len(number_of_participants) < num_scenarios:
        number_of_participants = [number_of_participants[0] for i in range(num_scenarios)]
    

    
    for i in range(num_scenarios):
        scenario = {
            # "action_space_string": "continuous",
            "number_of_participants": number_of_participants[i],
        }
        scenarios.append(scenario)

    with open(args.custom_config, "w") as f:
        json.dump(scenarios, f, indent=4)


import json
import argparse
import numpy as np
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

calculate here or input?

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
        default = "configs/pv_batt_.json",
        help="Path to custom config file")
    parser.add_argument(
        "--num_buildings",
        nargs='+',
        type=int,
        default = [10],
        help="List of num_buildings; if len < num_envs, use first as default")

    # pv parameters
    parser.add_argument(
        "--pv_mean",
        nargs='+',
        type=float,
        default = [0],
        help="List of pv means; if len < num_envs, use first as default")
    parser.add_argument(
        "--pv_var",
        nargs='+',
        type=float,
        default = [0],
        help="List of pv variances; if len < num_envs, use first as default")
    parser.add_argument(
        "--pv_bern",
        nargs='+',
        type=float,
        default = [0],
        help="List of pv Bernoulli probablities; if len < num_envs, use first as default")
    parser.add_argument(
        "--pv_mean_mean",
        type=float,
        default = -1,
        help="Mean of pv_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--pv_mean_var",
        type=float,
        default = -1,
        help="Variance of pv_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--pv_var_mean",
        type=float,
        default = -1,
        help="Mean of pv_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--pv_var_var",
        type=float,
        default = -1,
        help="Variance of pv_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--pv_bern_mean",
        type=float,
        default = -1,
        help="Mean of pv_berns, for generation. If < 0, use list of berns")
    parser.add_argument(
        "--pv_bern_var",
        type=float,
        default = -1,
        help="Variance of pv_berns, for generation. If < 0, use list of berns")
        
    # batt parameters
    parser.add_argument(
        "--batt_mean",
        nargs='+',
        type=float,
        default = [0],
        help="List of batt means; if len < num_envs, use first as default")
    parser.add_argument(
        "--batt_var",
        nargs='+',
        type=float,
        default = [0],
        help="List of batt variances; if len < num_envs, use first as default")
    parser.add_argument(
        "--batt_bern",
        nargs='+',
        type=float,
        default = [0],
        help="List of batt Bernoulli probablities (to set to 0); if len < num_envs, use first as default")
    parser.add_argument(
        "--batt_mean_mean",
        type=float,
        default = -1,
        help="Mean of batt_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--batt_mean_var",
        type=float,
        default = -1,
        help="Variance of batt_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--batt_var_mean",
        type=float,
        default = -1,
        help="Mean of batt_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--batt_var_var",
        type=float,
        default = -1,
        help="Variance of batt_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--batt_bern_mean",
        type=float,
        default = -1,
        help="Mean of batt_berns, for generation. If < 0, use list of berns")
    parser.add_argument(
        "--batt_bern_var",
        type=float,
        default = -1,
        help="Variance of batt_berns, for generation. If < 0, use list of berns")
    return parser
    

def expand_list(lst, length):
    if len(lst) < length:
        return [lst[0] for _ in range(length)]
    return lst

def return_normal_list(mean, var, n, lst, nonNegative):
    rv = np.array(lst)
    if mean >= 0 and var >= 0:
        rv = np.random.normal(mean, var, n)
    if nonNegative:
        rv[rv < 0] = 0
    return rv.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    scenarios = []
    num_scenarios = args.num_scenarios

    num_buildings = expand_list(args.num_buildings, num_scenarios)
    pv_mean = expand_list(args.pv_mean, num_scenarios)
    pv_var = expand_list(args.pv_var, num_scenarios)
    pv_bern = expand_list(args.pv_bern, num_scenarios)
    batt_mean = expand_list(args.batt_mean, num_scenarios)
    batt_var = expand_list(args.batt_var, num_scenarios)
    batt_bern = expand_list(args.batt_bern, num_scenarios)

    pv_mean = return_normal_list(args.pv_mean_mean, args.pv_mean_var, num_scenarios, pv_mean, False)
    pv_var = return_normal_list(args.pv_var_mean, args.pv_var_var, num_scenarios, pv_var, True)
    # Negative probabilities are set to 0, probabilities greater than 1 are treated equivalently to a probability of 1
    pv_bern = return_normal_list(args.pv_bern_mean, args.pv_bern_var, num_scenarios, pv_bern, True)

    batt_mean = return_normal_list(args.batt_mean_mean, args.batt_mean_var, num_scenarios, batt_mean, False)
    batt_var = return_normal_list(args.batt_var_mean, args.batt_var_var, num_scenarios, batt_var, True)
    # Negative probabilities are set to 0, probabilities greater than 1 are treated equivalently to a probability of 1
    batt_bern = return_normal_list(args.batt_bern_mean, args.batt_bern_var, num_scenarios, batt_bern, True) 

        

    
    for i in range(num_scenarios):
        pv = [0 if np.random.rand() < pv_bern[i] else np.random.normal(pv_mean[i], pv_var[i]) for _ in range(num_buildings[i])]
        batt = [0 if np.random.rand() < batt_bern[i] else np.random.normal(batt_mean[i], batt_var[i]) for _ in range(num_buildings[i])]
        
        # force all pv and batt values to be nonnegative and round to integer values
        pv = [max(0, round(i)) for i in pv]
        batt = [max(0, round(i)) for i in batt]

        scenario = {
            "pv": pv,
            "batt": batt
        }
        scenarios.append(scenario)

    with open(args.custom_scenario, "w") as f:
        json.dump(scenarios, f, indent=4)


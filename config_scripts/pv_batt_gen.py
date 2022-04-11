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
    parser.add_argument(
        "--pv_mean_mean",
        type=int,
        default = -1,
        help="Mean of pv_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--pv_mean_var",
        type=int,
        default = -1,
        help="Variance of pv_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--pv_var_mean",
        type=int,
        default = -1,
        help="Mean of pv_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--pv_mean_var",
        type=int,
        default = -1,
        help="Variance of pv_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--pv_bern_mean",
        type=int,
        default = -1,
        help="Mean of pv_berns, for generation. If < 0, use list of berns")
    parser.add_argument(
        "--pv_bern_var",
        type=int,
        default = -1,
        help="Variance of pv_berns, for generation. If < 0, use list of berns")
        
    # batt parameters
    parser.add_argument(
        "--batt_mean",
        nargs='+',
        type=int,
        default = [0],
        help="List of batt means; if len < num_envs, use first as default")
    parser.add_argument(
        "--batt_var",
        nargs='+',
        type=int,
        default = [0],
        help="List of batt variances; if len < num_envs, use first as default")
    parser.add_argument(
        "--batt_bern",
        nargs='+',
        type=int,
        default = [0],
        help="List of batt Bernoulli probablities (to set to 0); if len < num_envs, use first as default")
    parser.add_argument(
        "--batt_mean_mean",
        type=int,
        default = -1,
        help="Mean of batt_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--batt_mean_var",
        type=int,
        default = -1,
        help="Variance of batt_means, for generation. If < 0, use list of means")
    parser.add_argument(
        "--batt_var_mean",
        type=int,
        default = -1,
        help="Mean of batt_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--batt_mean_var",
        type=int,
        default = -1,
        help="Variance of batt_vars, for generation. If < 0, use list of vars")
    parser.add_argument(
        "--pv_bern_mean",
        type=int,
        default = -1,
        help="Mean of pv_berns, for generation. If < 0, use list of berns")
    parser.add_argument(
        "--pv_bern_var",
        type=int,
        default = -1,
        help="Variance of pv_berns, for generation. If < 0, use list of berns")
    return parser
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    scenarios = []
    num_scenarios = args.num_scenarios

    num_buildings = args.num_buildings
    if len(num_buildings) < num_scenarios:
        num_buildings = [num_buildings[0] for _ in range(num_scenarios)]
    pv_mean = args.pv_mean
    if len(pv_mean) < num_scenarios:
        pv_mean = [pv_mean[0] for _ in range(num_scenarios)]
    pv_var = args.pv_var
    if len(pv_var) < num_scenarios:
        pv_var = [pv_var[0] for _ in range(num_scenarios)]
    pv_bern = args.pv_bern
    if len(pv_mean) < num_scenarios:
        pv_bern = [pv_bern[0] for _ in range(num_scenarios)]
    batt_mean = args.batt_mean
    if len(batt_mean) < num_scenarios:
        batt_mean = [batt_mean[0] for _ in range(num_scenarios)]
    batt_var = args.batt_var
    if len(batt_var) < num_scenarios:
        batt_var = [batt_var[0] for _ in range(num_scenarios)]
    batt_bern = args.batt_bern
    if len(batt_mean) < num_scenarios:
        batt_bern = [batt_bern[0] for _ in range(num_scenarios)]
    

    pv_mean_mean = args.pv_mean_mean
    pv_mean_var = args.pv_mean_var
    if pv_mean_mean >= 0 and pv_mean_var >= 0:
        pv_mean = np.random.normal(pv_mean_mean, pv_mean_var, num_scenarios).tolist()
    pv_var_mean = args.pv_var_mean
    pv_var_var = args.pv_var_var
    if pv_var_mean >= 0 and pv_var_var >= 0:
        pv_var = np.random.normal(pv_var_mean, pv_var_var, num_scenarios).tolist()
    pv_bern_mean = args.pv_bern_mean
    pv_bern_var = args.pv_bern_var
    if pv_bern_mean >= 0 and pv_bern_var >= 0:
        pv_bern = np.random.normal(pv_bern_mean, pv_bern_var, num_scenarios).tolist()
    
    batt_mean_mean = args.batt_mean_mean
    batt_mean_var = args.batt_mean_var
    if batt_mean_mean >= 0 and batt_mean_var >= 0:
        batt_mean = np.random.normal(batt_mean_mean, batt_mean_var, num_scenarios).tolist()
    batt_var_mean = args.batt_var_mean
    batt_var_var = args.batt_var_var
    if batt_var_mean >= 0 and batt_var_var >= 0:
        batt_var = np.random.normal(batt_var_mean, batt_var_var, num_scenarios).tolist()
    batt_bern_mean = args.batt_bern_mean
    batt_bern_var = args.batt_bern_var
    if batt_bern_mean >= 0 and batt_bern_var >= 0:
        batt_bern = np.random.normal(batt_bern_mean, batt_bern_var, num_scenarios).tolist()

        

    
    for i in range(num_scenarios):
        pv = [0 if np.random.rand() < pv_bern[i] else np.random.normal(pv_mean[i], pv_var[i]) for _ in range(num_buildings[i])]
        batt = [0 if np.random.rand() < batt_bern[i] else np.random.normal(batt_mean[i], batt_var[i]) for _ in range(num_buildings[i])]
        scenario = {
            "pv": pv,
            "batt": batt
        }
        scenarios.append(scenario)

    with open(args.custom_config, "w") as f:
        json.dump(scenarios, f, indent=4)


import json
import argparse
import numpy as np
import os
import random

"""

    TL;DR:

    Scenarios are generated from normal distributions. For both pvsizes and battery_nums, there is a mean, variance, and
    bernoulli zeroing parameter. The parameters for these distributions can also be normally generated, or can be set to
    manually inputted values. Manual inputs (--pv_mean, etc) can be single values (used for all scenarios) or a list
    (used for corresponding scenario). To generate a parameter, set the mean and variance (alpha/beta for bern) for that
    parameter (--pv_mean_mean and --pv_mean_var to generate --pv_mean).



    HOW TO USE:


    This script is intended to generate pv_batt_scenarios. Currently, it only supports normally distributed scenarios.
    There are two levels of variation included here: between buildings within a scenario, and between scenarios. To model
    variation within a scenario, there is a mean and variance for each scenario, as well as a bernoulli probability for
    zeroing the value (no solar/batteries). To model variation between scenarios, there is the option to generate the
    scenario means and variances (and bernoulli probabilities) from normal distributions. For example, --pv_mean_mean and
    --pv_mean_var describe the mean and variance of a normal distribution, from which each scenario draws a value to use
    as its mean pv capacity per building.

    There are two methods of inputting values:

    One can change --pv_mean, --pv_var, and --pv_bern (and the batt versions) directly to define the properties of the
    normal distribution (mean, var) and zeroing probability (bern) for the pv (and batt) of each scenario. This can be a
    list of values equal to the number of scenarios, or a single value, in which case it is used for all scenarios. This
    method is useful for specific inputs, especially those that are intended to be the same for all scenarios (such as
    a constant zeroing bernoulli probability, for example).

    Alternatively, one can set --pv_mean_mean, --pv_mean_var, --pv_var_mean, --pv_var_var, --pv_bern_alpha, and
    --pv_bern_beta (and the corresponding batt parameters) to describe the distributions from which pv_means, pv_vars, 
    (normal) and pv_berns (beta) are drawn from for each scenario. For example, the pv_mean for each scenario is taken
    from a normal distribution with mean pv_mean_mean and variance pv_mean_var. This method is recommended for introducing
    variance between scenarios.

    
    Behavior of note:

        - All pv and battery numbers are rounded to the nearest integer and non-negative.
          This may result in greater than expected zero values for a given zeroing bernoulli probability if the mean/var
          would produce negative values.
        - Negative values are allowed for manually input values (--pv_mean, --pv_bern, etc) and will not error.
          However, negative variances will be set to zero. Note that non-negative final outputs are still enforced.
          (A negative mean with some variance may produce a smaller number of positive values and many zeros.)
        - Zeroing bernoulli probabilities can be any number, but values below zero will be treated as zero and values
          above 1 will be treated as 1.
        - In order to use distribution parameter generation (--pv_mean_mean, etc), both the mean and the variance
          components for that parameter must be set to a non-negative value (e.g. to generate pv_means, both --pv_mean_mean
          and --pv_mean_var must be >= 0). If this occurs, it will override the manually input values (--pv_means), if present.
          Otherwise, if both are not non-negative, it will default to the corresponding manual input.
        - Bernoulli zeroing probability Beta distribution probabilities must be positive rather than non-negative
        - It is entirely permissible to mix manual input (such as --pv_bern values) and generated parameters (such as
          --pv_var_mean and --pv_var_var).


    Config usage:
    
        - To use a custom pv_batt_scenario config, include --pv_batt_config_path 'YOUR_PATH_HERE' when running your experiment
        - If no config path is provided, will default to the original 6 pv_batt_scenarios
        - Scenarios can be accessed by using complex_pv_batt_scenario as an index on [0, num_scenarios - 1]


    Parameters:

        --num_scenarios:
            - number of different pv_batt_scenarios to generate
            - defaults to 3
            - scenarios will be indexed from 0 to (num_scenarios - 1)

        --custom_scenario:
            - path to where the custom config file will be created
            - defaults to 'configs/pv_batt_.json'

        --num_buildings:
            - number of buildings in each scenario
            - does not actually change the number of buildings used (this is controlled by the microgrid config),
              instead only changes the size of the pv and batt lists
            - defaults to 10
            - can be a list of different values, but varying this parameter currently has no significant effect

        --pv_mean:
            - means of the pvsizes distributions for each scenario
            - buildings in scenario i will have mean pv_mean[i] pvsizes
            - accepts a list of floats
            - if the number of input values is less than num_scenarios, uses the first value for all scenarios
            - ignores all extra values
            - defaults to all 0
        
        --pv_var:
            - variances of the pvsizes distributions for each scenario
            - buildings in scenario i will have variance pv_var[i] in pvsizes
            - accepts a list of floats
            - if the number of input values is less than num_scenarios, uses the first value for all scenarios
            - ignores all extra values
            - all negative values will be treated as 0
            - defaults to all 0

        --pv_bern:
            - bernoulli probability for zeroing a building's pvsize (probability of not having solar panels at all)
            - 0 = never zero out, 1 = zero every building
            - buildings in scenario i will have zeroed pvsizes with probability pv_bern[i]
            - accepts a list of floats
            - if the number of input values is less than num_scenarios, uses the first value for all scenarios
            - ignores all extra values
            - all negative values will be treated as 0 and all values > 1 will be treated as 1
            - defaults to all 0

        --pv_mean_mean:
            - mean of the pv_means across all scenarios
            - single float
            - must be non-negative to be used
            - requires pv_mean_var to also be set to be used
            - if used, will override pv_mean if present
            - default is -1 (off)
        
        --pv_mean_var:
            - variance of the pv_means across all scenarios
            - single float
            - must be non-negative to be used
            - requires pv_mean_mean to also be set to be used
            - if used, will override pv_mean if present
            - default is -1 (off)

        --pv_var_mean:
            - mean of the pv_vars across all scenarios
            - single float
            - must be non-negative to be used
            - requires pv_var_var to also be set to be used
            - if used, will override pv_var if present
            - default is -1 (off)
        
        --pv_var_var:
            - variance of the pv_vars across all scenarios
            - single float
            - must be non-negative to be used
            - requires pv_var_mean to also be set to be used
            - if used, will override pv_var if present
            - default is -1 (off)

        --pv_bern_alpha:
            - alpha parameter for the beta distribution from which pv_berns are drawn
            - single float
            - must be non-negative to be used
            - requires pv_bern_beta to also be set to be used
            - if used, will override pv_bern if present
            - default is -1 (off)
        
        --pv_bern_beta:
            - beta parameter for the beta distribution from which pv_berns are drawn
            - single float
            - must be positive to be used
            - requires pv_bern_alpha to also be set to be used
            - if used, will override pv_bern if present
            - default is -1 (off)

        --batt_mean, --batt_var, --batt_bern, --batt_mean_mean, --batt_mean_var, --batt_var_mean, --batt_var_var,
            --batt_bern_alpha, --batt_bern_beta:

            - behavior is the same as the corresponding pv entry above.


    Use Example:

    'python ./config_scripts/pv_batt_gen.py
      --num_scenarios 10
      --custom_scenario ./configs/pv_batt_configs/your_name_here.json
      --num_buildings 100
      --pv_mean_mean 100 --pv_mean_var 10
      --pv_var_mean 20 --pv_var_var 0
      --pv_bern_alpha 2 --pv_bern_beta 5
      --batt_mean 30 40 50 30 20 30 0 90 20 0
      --batt_var_mean 10 --batt_var_var 5
      --batt_bern 0.2'

    This command creates 10 different pv_batt_scenarios, which is saved to './configs/pv_batt_configs/your_name_here.json'.
    For each scenario, 100 buildings worth of values is generated (actual number of buildings used controlled by the microgrid
    itself). The mean of the pv distribution across buildings for each scenario is generated from Normal(100, 10), while the
    variance of the pv distribution across buildings for each scenario is generated from Normal(20, 0) [Note that this is
    functionally identical to simply setting pv_var to 20]. For each scenario, the probability of zeroing the pvsize for a given
    building is drawn from Beta(2, 5). The means of the battery_nums for each scenario is set to the ten numbers listed. The
    variance of the battery_num distribution across buildings for each scenario is generated from Normal(10, 5). A given building
    in every scenario has a 20% chance of having no batteries.

    When generating battery_nums for scenario index 0, a variance 'v' is sampled from Normal(10, 5) and used to create scenario
    index 0's battery_num distribution Normal(30, v). We repeat this 100 (num_buildings) times: with probability (1 - 0.2),
    sample a value from Normal(30, v) and keep this. Otherwise, with probability 0.2, instead output 0.
    

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
        help="Path to created custom config file")
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
        "--pv_bern_alpha",
        type=float,
        default = -1,
        help="Alpha parameter used to generate pv_berns, for generation. If < 0, use list of berns")
    parser.add_argument(
        "--pv_bern_beta",
        type=float,
        default = -1,
        help="Beta parameter used to generate pv_berns, for generation. If < 0, use list of berns")
        
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
        "--batt_bern_alpha",
        type=float,
        default = -1,
        help="Alpha parameter used to generate batt_berns, for generation. If <= 0, use list of berns")
    parser.add_argument(
        "--batt_bern_beta",
        type=float,
        default = -1,
        help="Beta parameter used to generate batt_berns, for generation. If <= 0, use list of berns")
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

def return_beta_list(alpha, beta, n, lst):
    rv = np.array(lst)
    if alpha > 0 and beta > 0:
        rv = np.random.beta(alpha, beta, n)
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
    pv_bern = return_beta_list(args.pv_bern_alpha, args.pv_bern_beta, num_scenarios, pv_bern)

    batt_mean = return_normal_list(args.batt_mean_mean, args.batt_mean_var, num_scenarios, batt_mean, False)
    batt_var = return_normal_list(args.batt_var_mean, args.batt_var_var, num_scenarios, batt_var, True)
    batt_bern = return_beta_list(args.batt_bern_alpha, args.batt_bern_beta, num_scenarios, batt_bern) 

        

    
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


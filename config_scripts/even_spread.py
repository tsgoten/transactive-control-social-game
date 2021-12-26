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
        default = "configs/even_spread.json",
        help="Path to custom config file"
    )
    return parser
if __name__ == '__main__':
    configs = []
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    num_envs = args.num_envs
    # Number of environments of each type
    num_cs_envs = args.num_envs // 2
    num_det_envs = args.num_envs - num_cs_envs
    det_choices = ["t", "l", "s"]
    # Allocate Curtail and Shift Environments
    for i in range(num_cs_envs):
        config = {
            "points_multiplier":random.choice(range(1, 20)),
            "shiftable_load_frac":random.uniform(0, 1),
            "curtailable_load_frac":random.uniform(0, 1),
            "shiftByHours":random.choice(range(1, 8), ),
            "maxCurtailHours":random.choice(range(1, 8),),
            "response_type_string": "cs"
        }
        configs.append(config)
    # Allocate Deterministic Environments (evenly spread among all choices)
    for i in range(num_det_envs):
        config = {
            "response_type_string": det_choices[i % len(det_choices)],
            "points_multiplier":random.choice(range(1, 20))
        }
        configs.append(config)
    with open(args.custom_config, "w") as f:
        json.dump(configs, f, indent=4)


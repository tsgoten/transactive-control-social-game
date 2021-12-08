import argparse
from gym import spaces

import numpy as np
import ray
from ray import tune
import ray.rllib.agents.sac.sac as sac

from gym_microgrid.envs.feudal_env import FeudalSocialGameHourwise


parser = argparse.ArgumentParser()
parser.add_argument()

upper_level_obs_space = spaces.Box(low = -np.inf, high = np.inf, shape = (20,), dtype = np.float64)
upper_level_action_space = spaces.Box(low = -1, high = 1, shape = (5,), dtype = np.float64)
lower_level_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
upper_level_action_space = spaces.Box(low = -1, high = 1, shape = (2,), dtype = np.float64)

policies = {"upper_level_agent": (
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

trainer = sac.SACTrainer(
    env=FeudalSocialGameHourwise, 
    config=config,
)

while True:
    print(trainer.train())
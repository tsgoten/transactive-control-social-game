import gym
from gym import spaces

import numpy as np
import pandas as pd 
import random

from gym_microgrid.envs.utils import price_signal
from gym_microgrid.envs.agents import *
from gym_microgrid.envs.reward import Reward
from gym_socialgame.envs.buffers import GaussianBuffer
from copy import deepcopy

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym_socialgame.envs.socialgame_env import SocialGameEnvRLLib
from gym_microgrid.envs.multiagent_env import MultiAgentSocialGameEnv


class FeudalSocialGameHourwise(MultiAgentEnv):
    def __init__(self, env_config):
        self.lower_level_env = FeudalSocialGameLowerHourEnv(env_config)
        #self.observation_space = self.lower_level_env._create_observation_space()
        #self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.price_in_state = True
        self.energy_in_state = True
        self.total_iter = 0

    def reset(self):
        ret = self.lower_level_env._get_observation()
        self.current_goals = np.zeros(5)
        return {"higher_level_agent": ret}
    
    def step(self, action_dict):
        if "higher_level_agent" in action_dict:
            print("executing higher level step")
            return self._high_level_step(action_dict["higher_level_agent"])
        else:
            print("executing lower level step")
            return self._low_level_step(np.concatenate(
                [action_dict["lower_level_agent_{}".format(i)] for i in range(5)]))

        # yesterdays energy: 10 dim, lower_ind, yesterday_energy[(lower_ind*2):(lower_ind*2+1)]

    def _upper_level_action_to_goal(self, action):
        """
        Purpose: map -1 to 1 --> 0 to 500
        """
        return (action + 1) * 250


    def _compute_lower_level_rewards(self, energy_tuple, goal):
        goal_tuple = [goal, goal]
        return np.linalg.norm(np.array(energy_tuple) - np.array(goal_tuple))

    def _high_level_step(self, action):
        print("higher level action")
        print(action)
        env_obs = self.lower_level_env._get_observation()
        
        obs = {"lower_level_agent_{}".format(i): np.concatenate((env_obs[2*i:(2*i + 2)], [action[i]])) for i in range(5)}

        ## previous goals 

        self.current_goals = self._upper_level_action_to_goal(action)

        rew = {"lower_level_agent_{}".format(i): self._compute_lower_level_rewards(
            env_obs[(10 + 2*i) : (10 + (2*i + 2))], 
            self.current_goals[i]
        ) for i in range(5)}

        done = {"__all__": False}

        print("higher level obs")
        print(obs)

        self.total_iter += 1 
        
        return obs, rew, done, {}

    def _low_level_step(self, action): 
        f_obs, f_rew, f_done, _ = self.lower_level_env.step(action)
        
        print("lower level obs")
        obs = {"higher_level_agent": f_obs}
        rew = {"higher_level_agent": f_rew}
        done = {"__all__": f_done}

        print(obs)
        return obs, rew, done, {}

class FeudalSocialGameLowerHourEnv(SocialGameEnvRLLib):
    def __init__(self, env_config):
        super().__init__(env_config)
        print("Initialized RLLib lower agent class")
    
    def _create_observation_space(self):
        """
        Purpose: Returns the observation space.

        # 2 dimensions for the previous energy use under the two hours this agent focuses on
        # 1 dimension for the action of the manager (total energy)

        Args:
            None

        Returns:
            Action Space for environment based on action_space_str
        """

        print("initialized lower level agent observation space")
        dim = 3
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


    def set_total_demand_attribute(self, total_demand):
        self.total_demand = total_demand
        return
    
    def set_upper_level_command(self, upper_level_command):
        self.upper_level_command = upper_level_command
        return


    def step(self, action):
        """
        Purpose: A single macro step for all 5 lower level agents 

        Args:
            Action: a 10 dim vector of energy by all lower level agents 

        Returns:
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)
        """

        self.action = action

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        points = self._points_from_action(action)

        energy_consumptions = self._simulate_humans(points)
        self.energy_consumptions = energy_consumptions
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)
        observation = self._get_observation()
    
        done = True

        return observation, reward, done, {}


    def _get_observation(self):
        print("get_observation in the lower_level_env")
        self.prev_price = self.prices[ (self.day - 1) % 365]
        next_price = self.prices[self.day]

        if self.bin_observation_space:
            self.prev_energy = np.round(self.prev_energy, -1)

        next_observation = np.array([])

        self.price_in_state = self.energy_in_state = True
        if self.price_in_state:
            next_observation = np.concatenate((next_observation, next_price))

        if self.energy_in_state:
            next_observation = np.concatenate((next_observation, self.prev_energy), dtype=np.float32)

        return next_observation

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

class FeudalSocialGameUpper2HourAgent(SocialGameEnvRLLib):
    def __init__(self, env_config, num_lower_agents=5):
        super().__init__(env_config)
        self.num_lower_agents = num_lower_agents
        print("Initialized RLLib upper agent class")


    def _create_observation_space(self):
        """
        Purpose: Returns the observation space.
        dim is 10 for previous days energy usage +  10 for prices

        Args:
            None

        Returns:
            Action Space for environment based on action_space_str
        """

        dim = 20 # self.hours_in_day*np.sum([self.price_in_state, self.energy_in_state])
        #TODO: Normalize obs_space !
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float64)


    def _create_action_space(self):
        """
        Purpose: first cut, propose a discrete energy amount that each agent can target

        Args:
            None

        Returns:
            Action Space for environment based on action_space_str

        Note: Multidiscrete refers to a 10-dim vector where each action {0,1,2} represents Low, Medium, High points respectively.
        We pose this option to test whether simplifying the action-space helps the agent.
        """

        #Making a symmetric, continuous space to help learning for continuous control (suggested in StableBaselines doc.)
        # if self.action_space_string == "continuous":
        #     return spaces.Box(low=-1, high=1, shape=(self.action_length,), dtype=np.float32)

        return spaces.Box(low=-1, high=1, shape=(self.num_lower_agents,), dtype=np.float32)
    
    def set_yesterdays_points(self, yesterday_points):
        self.yesterday_points = yesterday_points
        return

    def step(self, action):
        """
        Purpose: Takes a step in the environment

        Args:
            Action: 10-dim vector detailing player incentive for each hour (8AM - 5PM)

        Returns:
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        Exceptions:
            raises AssertionError if action is not in the action space
        """
        self.two_hour_energy_blocks = action

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1

        done = self.curr_iter > 0

        # set_yesterday_points() must be called first 
        energy_consumptions = self._simulate_humans(self.yesterday_points)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)

        if self.use_smirl:
            self.buffer.add(observation)

        info = {}

        return observation, reward, done, info


    def _get_observation(self):
        prev_price = self.prices[ (self.day - 1) % 365]
        next_price = self.prices[self.day]
        next_observation = np.concatenate((next_price, self.prev_energy))

        return next_observation


class FeudalSocialGameLower3HourAgent(SocialGameEnvRLLib):
    
    def __init__(self, env_config, number, num_lower_agents=5):
        super().__init__(env_config)
        self.num = number
        print("Initialized RLLib upper agent class")
    
    def _create_action_space(self):
        """
        Purpose: first cut, propose a discrete energy amount that each agent can target

        Args:
            None

        Returns:
            Action Space for environment based on action_space_str

        Note: Multidiscrete refers to a 10-dim vector where each action {0,1,2} represents Low, Medium, High points respectively.
        We pose this option to test whether simplifying the action-space helps the agent.
        """

        #Making a symmetric, continuous space to help learning for continuous control (suggested in StableBaselines doc.)
        # if self.action_space_string == "continuous":
        #     return spaces.Box(low=-1, high=1, shape=(self.action_length,), dtype=np.float32)

        return spaces.Box(low = -1, high =1, shape = (10 / self.num_lower_agents,), dtype=np.float32)
    
    def _create_observation_space(self):
        """
        Purpose: Returns the observation space.
        # TODO: Jonathan, what do you think? 

        # 2 dimensions for the previous energy use under the two hours this agent focuses on
        # 1 dimension for the action of the manager (total energy)

        Args:
            None

        Returns:
            Action Space for environment based on action_space_str
        """

        dim = 3
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float64)


    def set_total_demand_attribute(self, total_demand):
        self.total_demand = total_demand
        return
    
    def set_upper_level_command(self, upper_level_command):
        self.upper_level_command = upper_level_command
        return



    def step(self, action):
        """
        Purpose: Takes a step in the environment

        Args:
            Action: 2-dim vector detailing player incentive for each hour (8AM - 5PM)

        Returns:
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        Exceptions:
            raises AssertionError if action is not in the action space
        """
        self.action = action

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        self.total_iter +=1

        done = self.curr_iter > 0

        demand_responsible = self.total_demand[self.num*2:(self.num*2 + 1)]

        reward = np.abs(self.upper_level_command - demand_responsible)
        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same

        observation = self._get_observation()

        if self.use_smirl:
            self.buffer.add(observation)

        info = {}

        return observation, reward, done, info
 

    def _get_observation(self):
        prev_price = self.prices[ (self.day - 1) % 365]
        next_price = self.prices[self.day]

        if self.bin_observation_space:
            self.prev_energy = np.round(self.prev_energy, -1)

        next_observation = np.array([])

        if self.price_in_state:
            next_observation = np.concatenate((next_observation, next_price))

        if self.energy_in_state:
            next_observation = np.concatenate((next_observation, self.prev_energy))

        return next_observation

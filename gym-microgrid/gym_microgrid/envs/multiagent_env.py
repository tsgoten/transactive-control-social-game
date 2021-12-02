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
from gym_microgrid.envs.microgrid_env import MicrogridEnvRLLib

class BaseMultiAgentEnv(MultiAgentEnv):
    @property
    def last_energy_reward(self):
        return {str(i): env.last_energy_reward for i, env in enumerate(self.envs)}

    @property
    def last_energy_cost(self):
        return {str(i): env.last_energy_cost for i, env in enumerate(self.envs)}
        
    @property
    def last_energy_cost(self):
        return {str(i): env.last_smirl_reward for i, env in enumerate(self.envs)}

    @property
    def use_smirl(self):
        return {str(i): env.use_smirl for i, env in enumerate(self.envs)}

    def step(self, action_dict):
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        info_dict = {}
        all_ = True
        self.total_iter += 1
        for i, action in action_dict.items():
            observation, reward, done, info = self.envs[int(i)].step(action)
            obs_dict[i] = observation
            rew_dict[i] = reward
            done_dict[i] = done
            info_dict[i] = info
            if done:
                all_ = all_ and done
            #observation = np.concat(observation, np.array([self.curr_env_id]), axis=-1)
        info_dict["__all__"] = all_
        return obs_dict, rew_dict, info_dict, done_dict

    def _get_observation(self):
        ret =  {str(i): self.envs[i]._get_observation() for i in range(len(self.envs))}
        #breakpoint()
        return ret

    def reset(self):
        """ Resets the environment on the current day """
        ret = self._get_observation()
        return ret

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def check_valid_init_inputs(self, complex_batt_pv_scenarios, num_inner_steps):
        assert num_inner_steps > 0, "need a positive num_inner_steps"
        assert len(complex_batt_pv_scenarios) > 0, "at least one scenario must be provided"

class MultiAgentMicrogridEnv(BaseMultiAgentEnv):
    def __init__(self, env_config):
        """
        MicrogridEnv for an agent determining incentives in a social game.

        Note: One-step trajectory (i.e. agent submits a 24-dim vector containing transactive price for each hour of each day.
            Then, environment advances one-day and agent is told that the episode has finished.)

        Args:
            action_space_string: (String) either "continuous" or "multidiscrete"
            number_of_participants: (Int) denoting the number of players in the social game (must be > 0 and < 20)
            one_day: (Int) in range [-1,365] denoting which fixed day to train on .
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,365] = Day of the Year
            energy_in_state: (Boolean) denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: (Boolean) denoting whether (or not) to append yesterday's price signal to the state
        """
        self.check_valid_init_inputs(env_config["scenarios"], env_config["num_inner_steps"])
        self.complex_batt_pv_scenarios = env_config["scenarios"]
        self.num_inner_steps = env_config["num_inner_steps"] 
        self.total_iter = 0
        
        self.configs = [deepcopy(env_config) for _ in self.complex_batt_pv_scenarios]
        for i, config in enumerate(self.configs):
            config["complex_batt_pv_scenario"] = int(self.complex_batt_pv_scenarios[i])
        self.envs = [MicrogridEnvRLLib(config) for config in self.configs]
        # self.envs = [SocialGameEnvRLLib(config) for config in self.configs]
        #WARNING: THESE WILL NOT WORK IF NOT ALL ENVS HAVE THE SAME OBS/ACTION SPACE
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

class MultiAgentSocialGameEnv(BaseMultiAgentEnv):
    def __init__(self, env_config):
        """
        MultiAgent implementation of SocialGame. TODO: Add descreption. 
        """
        # TODO: Modify so it's not just a copy of the same agent
        self.configs = [deepcopy(env_config) for _ in range(3)]
        self.total_iter = 0
        self.envs = [SocialGameEnvRLLib(config) for config in self.configs]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

class FeudalSocialGameEnv(MultiAgentSocialGameEnv):
    def __init__(self, env_config):
        """
        MultiAgent implementation of a feudal approach to SocialGame. TODO: Add descreption. 
        """
        # TODO: Modify so it's not just a copy of the same agent
        self.configs = [deepcopy(env_config) for _ in range(3)]
        self.total_iter = 0
        self.envs = [SocialGameEnvRLLib(config) for config in self.configs]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
from copy import deepcopy
import gym
from gym import spaces

import numpy as np
import pandas as pd 
import random

from gym_microgrid.envs.utils import price_signal
from gym_microgrid.envs.agents import *
from gym_microgrid.envs.reward import Reward

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym_socialgame.envs.socialgame_env import SocialGameEnvRLLib
from gym_microgrid.envs.feudal_env import (FeudalMicrogridEnvHigherAggregator, FeudalMicrogridEnvLowerAggregator)
from gym_microgrid.envs.microgrid_env import MicrogridEnvRLLib
from gym_microgrid.envs.multiagent_env import MultiAgentMicrogridEnv

import pdb

class FeudalMicrogridEnvOnlyLowerBaselineEnv(FeudalMicrogridEnvHigherAggregator):
    """
    Baseline 2. 

        Baseline 1: we freeze the upper level actions. Suggestion: tweak the hierarchical env. 
            Tweak obs, maybe reset, and the way the lower level step handles upper level actions. 
            This will have 6 independent lower level agents. 
        
        Baseline 2: we freeze (i.e. disregard) the lower level actions. We need copies of the lower level envs, 
            but not necessarily the agents. (We'll work on this later!)
    """
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.sell_price_grid = np.zeros(24)
        self.buyprice_grid = np.zeros(24)
        self.total_iter = 0
        print("ended init")

    def _set_lower_level_attributes(self, buyprice_grid, sellprice_grid):
        for agent in range(6):
            self.lower_level_agent_dict[
                f"lower_level_agent_{agent}"].higher_level_buy_price = (
                    buyprice_grid
                )
            self.lower_level_agent_dict[
                f"lower_level_agent_{agent}"].higher_level_sell_price = (
                    sellprice_grid
                )
            self.lower_level_agent_dict[
                f"lower_level_agent_{agent}"].grid_buy_price = (
                    buyprice_grid
                )
            self.lower_level_agent_dict[
                f"lower_level_agent_{agent}"].grid_sell_price = (
                    sellprice_grid
                )
        return

    def step(self, action_dict):
        """
        want to obs{}, rew{}, done info with the lower level agents filled in. 
        """
        self.total_iter += 1
        self.buyprice_grid = self.buyprices_grid[(self.day + 1) % 365] 
        self.sell_price_grid = self.sellprices_grid[(self.day + 1) % 365]

        self._set_lower_level_attributes(self.buyprice_grid, self.sell_price_grid)

        obs, rew, done, info = self._low_level_step(action_dict)

        obs = {key: obs[key] for key in obs if key != 'higher_level_agent'}
        rew = {key: rew[key] for key in rew if key != 'higher_level_agent'}
        done = {key: done[key] for key in done if key != 'higher_level_agent'}

        self.last_energy_rewards = rew
        self.last_energy_costs =  {
            f"lower_level_agent_{i}": 
            self.lower_level_agent_dict[f"lower_level_agent_{i}"].money_from_prosumers
            for i in range(6)
        }
        
        self.batt_stats = {
            f"lower_level_agent_{i}": {
                "discharge_caps": self.lower_level_agent_dict[f"lower_level_agent_{i}"].battery_discharges_cap_today,
                "discharges": self.lower_level_agent_dict[f"lower_level_agent_{i}"].battery_discharges_times_today
            }
            for i in range(6)
        }
        print(f"batt_stats for agent 1 are: {self.batt_stats['lower_level_agent_1']}")

        return obs, rew, done, {}
    
    def reset(self):
        self.last_energy_rewards = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_costs = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_goals = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        
        return {f"lower_level_agent_{i}": np.concatenate((
                self.lower_level_agent_dict[f"lower_level_agent_{i}"].generation_tomorrow,
                self.buyprice_grid, # buyprice_grid
                self.sell_price_grid, # sellprice_grid_tomorrow
                self.buyprice_grid,
                self.sell_price_grid,
                self.lower_level_agent_dict[f"lower_level_agent_{i}"].prev_energy
            )) 
            for i in range(6)
        }
    

# class FeudalMicrogridOnlyUpperBaselineEnv():

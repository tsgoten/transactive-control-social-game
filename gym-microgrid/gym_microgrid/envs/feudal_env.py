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
from gym_microgrid.envs.multiagent_env import MultiAgentMicrogridEnv

import pdb

class FeudalSocialGameHourwise(MultiAgentEnv):
    def __init__(self, env_config):
        self.lower_level_env = FeudalSocialGameLowerHourEnv(env_config)
        #self.observation_space = self.lower_level_env._create_observation_space()
        #self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.price_in_state = True
        self.energy_in_state = True
        self.total_iter = 0
        
        self.last_energy_rewards = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_rewards["higher_level_agent"] = 0
        self.last_energy_costs = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_costs["higher_level_agent"] = 0
        self.last_goals = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_goals["higher_level_agent"] = 0
        self.lower_level_reward_type = env_config["lower_level_reward_type"]

    def reset(self):
        self.last_energy_rewards = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_rewards["higher_level_agent"] = 0
        self.last_energy_costs = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_costs["higher_level_agent"] = 0
        self.last_goals = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_goals["higher_level_agent"] = 0
        ret = self.lower_level_env._get_observation()
        self.current_goals = np.zeros(5)
        return {"higher_level_agent": ret}
    
    def step(self, action_dict):
        if "higher_level_agent" in action_dict:
            return self._high_level_step(action_dict["higher_level_agent"])
        else:
            return self._low_level_step(np.concatenate(
                [action_dict["lower_level_agent_{}".format(i)] for i in range(5)]))

        # yesterdays energy: 10 dim, lower_ind, yesterday_energy[(lower_ind*2):(lower_ind*2+1)]

    def _higher_level_action_to_goal(self, action):
        """
        Purpose: map -1 to 1 --> 0 to 500
        """
        return (action + 1) * 250


    def _compute_lower_level_rewards(
        self, 
        energy_tuple, 
        yesterday_energy_tuple,
        goal, 
        type = "directional"):

        energy_tuple = np.array(energy_tuple)
        yesterday_energy_tuple = np.array(yesterday_energy_tuple)
        goal_tuple = np.repeat(goal/2, 2)

        if type == "directional":
            energy_diff = energy_tuple - yesterday_energy_tuple
            num = np.multiply(energy_diff, goal_tuple)
            denom = np.linalg.norm(energy_diff) * np.linalg.norm(goal_tuple)
            return  num / denom
        elif type == "l1":
            return -np.sum(np.abs(energy_tuple - goal_tuple))
        elif type == "l2":
            return  -np.linalg.norm(np.array(energy_tuple) - np.array(goal_tuple))
        else:
            raise NotImplementedError("Wrong lower level reward type specified.")
        """
        np.abs((e_1 + e_2) - goal) # this is just absolute value
        norml1((e_1,e_2) - (goal, goal)) 
        norml1((e_1,e_2) - (goal, goal)) <-- sqrt((e_1 - goal) ^2 + (e_2 - goal)^2)
        
        
        e = f(p, b) 
            10 hour energy vector e 
            [e_1, e_2, ..., e_10]
        """


    def _compute_lower_level_costs(self, energy_tuple, price_tuple):
        total_agent_energy_cost = np.sum(energy_tuple * price_tuple)
        return total_agent_energy_cost


    def _high_level_step(self, action):
        print("higher level action")
        print(action)
        env_obs = self.lower_level_env._get_observation()
        
        obs = {
            "lower_level_agent_{}".format(i): np.concatenate(
                (env_obs[2*i:(2*i + 2)], 
                env_obs[(10 + 2*i) : (10 + (2*i + 2))], 
                [action[i]])) 
            for i in range(5)} ## TODO: this is being set twice, should this matter? 

        ## previous goals 

        self.current_goals = self._higher_level_action_to_goal(action)
        for i in range(5):
            self.last_goals["lower_level_agent_{}".format(i)] = self.current_goals[i]

        self.last_goals["higher_level_agent"] = 0
        rew = {"lower_level_agent_{}".format(i): 0 for i in range(5)}

        done = {"__all__": False}

        print("higher level obs")
        print(obs)

        self.total_iter += 1 
        
        return obs, rew, done, {}

    def _low_level_step(self, action): 
        yesterday_obs = self.lower_level_env._get_observation()
        yesterday_energy = yesterday_obs[10:]
        f_obs, f_rew, f_done, _ = self.lower_level_env.step(action) ### TODO: not the action we think... I think 
        
        print("lower level obs")
        
        env_obs = self.lower_level_env._get_observation()
        
        # TODO: this commented out code is left as a test to see if the env is functioning right
        # obs = {"lower_level_agent_{}".format(i): np.concatenate(   ## TODO: this happens twice? 
        #     (
        #         env_obs[2*i:(2*i + 2)],
        #         env_obs[(10 + 2*i) : (10 + (2*i + 2))], 
        #         [action[i]]) ### TODO: here action is treated as a goal
        #     ) 
        #     for i in range(5)}

        obs = {"lower_level_agent_{}".format(i): np.zeros(5) for i in range(5)}
        obs.update({"higher_level_agent": f_obs})
        
        rew = {"lower_level_agent_{}".format(i): self._compute_lower_level_rewards(
            energy_tuple=f_obs[(10 + 2*i) : (10 + (2*i + 2))], 
            yesterday_energy_tuple=yesterday_energy[(2*i) : (2*i + 2)],
            goal=self.current_goals[i],
            type = self.lower_level_reward_type
        ) for i in range(5)}
        rew.update({"higher_level_agent": f_rew})
        done = {"__all__": f_done}

        self.last_energy_rewards = rew
        
        self.last_energy_costs = {"lower_level_agent_{}".format(i): self._compute_lower_level_costs(
            f_obs[(10 + 2*i) : (10 + (2*i + 2))],
            f_obs[(2*i) : (2*i + 2)]
        ) for i in range(5)}
        
        self.last_energy_costs["higher_level_agent"] = self.lower_level_env.last_energy_cost

        return obs, rew, done, {}

class FeudalSocialGameLowerHourEnv(SocialGameEnvRLLib):
    def __init__(self, env_config):
        super().__init__(env_config)
        print("Initialized RLLib lower agent class")
    
    def _create_observation_space(self):
        """
        Purpose: Returns the observation space.

        # 2 dimensions for today's energy price under the two hours this agent focuses on
        # 2 dimensions for the previous day's energy use under the two hours the agent focuses on
        # 1 dimension for the action of the manager (total energy)

        Args:
            None

        Returns:
            Observation space 
        """

        print("initialized lower level agent observation space")
        dim = 5
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


    def set_total_demand_attribute(self, total_demand):
        self.total_demand = total_demand
        return
    
    def set_higher_level_command(self, higher_level_command):
        self.higher_level_command = higher_level_command
        return


    def step(self, action):
        """
        Purpose: A single macro step for all 5 lower level agents 

        Args:
            Action: a 10 dim vector of energy by all lower level agents  ## TODO: FALSE!

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
        self.prev_energy = np.abs(energy_consumptions["avg"]) ## TODO: should be more rigorous about checking negative energy!
        reward = self._get_reward(prev_price, energy_consumptions, reward_function = self.reward_function)
        print("reward in lower step")
        print(reward)
        self.last_energy_cost = np.sum(prev_price * np.abs(energy_consumptions["avg"]))
        print("self.last_energy_cost in lower step")
        print(self.last_energy_cost)
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



class FeudalMicrogridEnvHigherAggregator(MultiAgentEnv):
    """
    Higher level agent 

    state space: (grid buy and sell prices, energy_demand_grid_1_yesterday, energy_demand_grid_2_yesterday, ..., energy_demand_6_yesterday)
    action: (aggregator_buy_price, aggregator_sell_price) # 48-d vector 
    reward: (buy_price^T .negative hours + sell_price^T.positive hours) + (lower_level_profit_1) +... 

    lower_level_energy_consumptions can be broken up into positive and negative, i.e.:
        positive:
        1: [0,3,4,...]
        2: [3,4,0,...]
        3: [2,0,0,...]
        sum: [5,7,4,...]

        negative:
        1: [-1,0,0,...]
        2: [0,0,-4,...]
        3: [0,-3,-6,...]
        sum: [-1, -3, -10]
    """

    def __init__(self, env_config):
        super().__init__()
        self.higher_level_aggregator_buyprice = np.zeros(24)
        self.higher_level_aggregator_sellprice = np.zeros(24)
        self.last_energy_rewards = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_rewards["higher_level_agent"] = 0
        self.last_energy_costs = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_costs["higher_level_agent"] = 0
        self.last_goals = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_goals["higher_level_agent"] = 0

        self.day = 0
        self.day_length = 24 

        self.total_iter = 0

        self.buyprices_grid, self.sellprices_grid = self._get_prices()
        # self.prices = self.buyprices_grid #Initialise to buyprices_grid

        self.lower_level_agent_dict = {
            f"lower_level_agent_{i}": 
            FeudalMicrogridEnvLowerAggregator(env_config, battery_pv_scenario = i) 
            for i in range(6)
        }
        print("ended init")
    

    def _get_prices(self):
        """
        Purpose: Get grid price signals for the entire year (PG&E commercial rates)

        Args:
            None

        Returns: Two arrays containing 365 price signals, where array[day_number] = grid_price for day_number 
        One each for buyprice and sellprice: sellprice set to be a fraction of buyprice

        """

        buy_prices = []
        sell_prices = []


        # Read PG&E price from CSV file. Index starts at 5 am on Jan 1, make appropriate adjustments. For year 2012: it is a leap year
        # price = pd.read_csv('/Users/utkarshapets/Documents/Research/Optimisation attempts/building_data.csv')[['Price( $ per kWh)']]
        price = np.squeeze(pd.read_csv('gym-microgrid/gym_microgrid/envs/building_data.csv')[['Price( $ per kWh)']].values)

        for day in range(0, 365):
            buyprice = price[day*self.day_length+19 : day*self.day_length+19+24]
            sellprice = 0.6*buyprice
            buy_prices.append(buyprice)
            sell_prices.append(sellprice)

        return buy_prices, sell_prices
    
    def reset(self):
        print("reset")
        self.last_energy_rewards = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_rewards["higher_level_agent"] = 0
        self.last_energy_costs = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_energy_costs["higher_level_agent"] = 0
        self.last_goals = {"lower_level_agent_{}".format(i): 0 for i in range(5)}
        self.last_goals["higher_level_agent"] = 0

        # this is the higher level agent's observation 
        ret = self._get_observation() ## TODO: set day = 0? 
      
        return {"higher_level_agent": ret.tolist()}

    
    def _get_observation(self):
        print("_get_observation function")
        buyprice_grid_tomorrow = self.buyprices_grid[(self.day + 1) % 365] 
        sell_price_grid_tomorrow = self.sellprices_grid[(self.day + 1) % 365]

        obs =  np.append(
            np.hstack(
                (
                    buyprice_grid_tomorrow,
                    sell_price_grid_tomorrow,)), 
            np.array(
                    [
                        self.lower_level_agent_dict[
                            f"lower_level_agent_{i}"].prev_energy 
                        for i in range(6)
                    ] 
                ))#.astype(np.float32)
        print(obs.shape)
        return obs

    def step(self, action_dict):
        print("general step function")
        if "higher_level_agent" in action_dict:
            return self._high_level_step(action_dict["higher_level_agent"])
        else:
            return self._low_level_step(action_dict)


    def _high_level_step(self, action):
        """
        action is 24-dim sell price, then 24-dim buy price
        """
        print("higher level step")
        # set observation: 
        higher_level_obs = self._get_observation()

        buyprice_grid = higher_level_obs[:24]
        sellprice_grid = higher_level_obs[24:48]

        self.higher_level_aggregator_buyprice, self.higher_level_aggregator_sellprice = (
            self._price_from_action(
                action, 
                buyprice_grid,
                sellprice_grid
            ))
        obs = {
            f"lower_level_agent_{i}": np.concatenate((
                self.lower_level_agent_dict[f"lower_level_agent_{i}"].generation_tomorrow,
                buyprice_grid, # buyprice_grid_tomorrow
                sellprice_grid, # sellprice_grid_tomorrow
                self.higher_level_aggregator_buyprice,
                self.higher_level_aggregator_sellprice,
                self.lower_level_agent_dict[f"lower_level_agent_{i}"].prev_energy
            ))
            for i in range(6)
        }

        ## setting environmental variables in the lower envs so that
        # I'm sure that there are the higher buy prices being set  
        for agent in range(6):
            self.lower_level_agent_dict[
                f"lower_level_agent_{agent}"].higher_buy_price = (
                    self.higher_level_aggregator_buyprice
                )
            self.lower_level_agent_dict[
                f"lower_level_agent_{agent}"].higher_sell_price = (
                    self.higher_level_aggregator_sellprice
                )

        self.total_iter += 1

        rew = {f"lower_level_agent_{i}": 0 for i in range(6)}
        done = {"__all__": False}

        return obs, rew, done, {}

    def _low_level_step(self, action):
        print("made it to lower level step")
        obs = {}
        rew = {}
        done = {}
        info = {}

        for agent in range(6):
            self.lower_level_agent_dict[f"lower_level_agent_{agent}"].day = self.day
            (obs[f"lower_level_agent_{agent}"], 
            rew[f"lower_level_agent_{agent}"],
            done[f"lower_level_agent_{agent}"],
            info[f"lower_level_agent_{agent}"]) = (
                self.lower_level_agent_dict[f"lower_level_agent_{agent}"].step(
                    action[f"lower_level_agent_{agent}"]) # TODO: is it stored like this? 
            )

        higher_level_obs = self._get_observation()
        obs.update({"higher_level_agent": higher_level_obs})

        # calculate reward 
        microgrid_energy_consumptions = {f"lower_level_agent_{agent}":
            self.lower_level_agent_dict[f"lower_level_agent_{agent}"].prev_energy 
            for agent in range(6)}

        higher_level_profit = self._calculate_higher_level_reward(
            higher_level_obs[:24], #buyprice_grid_tomorrow
            higher_level_obs[24:48],
            self.higher_level_aggregator_buyprice,
            self.higher_level_aggregator_sellprice,
            microgrid_energy_consumptions
        )

        lower_level_profit_total = sum([rew[f"lower_level_agent_{i}"] for i in range(6)])

        print("higher level profit")
        print(higher_level_profit)
        print("lower level profit")
        print(lower_level_profit_total)

        rew["higher_level_agent"] = higher_level_profit + lower_level_profit_total

        done = {"__all__": True}
        self.day +=1 # TODO does this go here or in higher level step? 

        return obs, rew, done, {}
    
    def _price_from_action(self, action, buyprice_grid, sellprice_grid):
        """
        Purpose: Convert agent actions that lie in [-1,1] into transactive price (conversion is for multidiscrete setting)

        Args:
            Action: 24-dim vector corresponding to action for each hour

        Returns: Price: 24-dim vector of transactive prices
        """
        
        # Continuous space is symmetric [-1,1], we map to -> [sellprice_grid,buyprice_grid] 
        # -1 -> sellprice. 1 -> buyprice

        midpoint_price = (buyprice_grid + sellprice_grid)/2
        diff_grid = buyprice_grid - sellprice_grid
        scaled_diffs_bp = np.multiply(action[0:24], diff_grid)/2 # Scale to fit difference at each hour
        scaled_diffs_sp = np.multiply(action[24:], diff_grid)/2 # Scale to fit difference at each hour
        buyprice = scaled_diffs_bp + midpoint_price
        sellprice = scaled_diffs_sp + midpoint_price

        return buyprice, sellprice


    def _calculate_higher_level_reward(
        self, 
        buyprice_grid, 
        sellprice_grid, 
        higher_aggregator_buyprice, 
        higher_aggregator_sellprice, 
        microgrids_energy_consumptions
        ):

        total_consumption = np.sum([i for i in microgrids_energy_consumptions.values()], axis=0)

        money_to_utility = (
            np.dot(np.maximum(0, total_consumption), buyprice_grid) + 
            np.dot(np.minimum(0, total_consumption), sellprice_grid))

        money_from_prosumers = 0

        for prosumerName, consumptions in microgrids_energy_consumptions.items():
            money_from_prosumers += (
                (np.dot(np.maximum(0, consumptions), higher_aggregator_buyprice) + 
                np.dot(np.minimum(0, consumptions), higher_aggregator_sellprice))
            )

        total_reward = money_from_prosumers - money_to_utility

        print(f"total_reward = {total_reward}")
        
        return total_reward


class FeudalMicrogridEnvLowerAggregator(MicrogridEnvRLLib):
    """
    Lower level agent:

    state space: (grid buy and sell prices, aggregator buy and sell prices, energy_demand_grid_yesterday)
    action: (lower_aggregator_buy_price, lower_aggregator_sell_price) # 48-d vector 

    optimal_external_buy_price = max (higher_level_buy_price, utility_buy_price)
    optimal_external_sell_price = max( upperlevel_sell_price, utility_sell_price)

    reward: (buy_price^T .negative hours + optimal_external_sell_price ^T . positive_hours) - (sell_price^T.positive hours + optimal_external_buy_price^T . negative_hours)
    """

    def __init__(self, env_config, battery_pv_scenario):
        super().__init__(env_config)
        self.prev_energy = np.random.sample(24)
        self.complex_batt_pv_scenario = battery_pv_scenario
        self.prosumer_dict = self._create_agents()
        self.reward_function = "profit_maximizing"
        self.higher_level_sell_price = np.zeros(24)
        self.higher_level_buy_price = np.zeros(24)
        self.generation_tomorrow = np.zeros(24)
    
    def _create_observation_space(self):
        dim = 24 + (24 + 24) + (24 + 24) + 24
        return spaces.Box(
            low = -np.inf, 
            high = np.inf, 
            shape = (dim,), 
            dtype = np.float64
            )


    def _get_observation(self):
    
        prev_energy = self.prev_energy
        generation_tomorrow = self.generation[(self.day + 1)%365] 
        buyprice_grid_tomorrow = self.buyprices_grid[(self.day + 1)%365] 
        sellprice_grid_tomorrow = self.sellprices_grid[self.day]

        noise = np.random.normal(loc = 0, scale = 50, size = 24) ## TODO: get rid of this if not doing well
        generation_tomorrow_nonzero = (generation_tomorrow > abs(noise)) # when is generation non zero?
        generation_tomorrow += generation_tomorrow_nonzero* noise # Add in Gaussian noise when gen in non zero

        self.generation_tomorrow

        return np.concatenate(
            (
                generation_tomorrow, 
                buyprice_grid_tomorrow,
                sellprice_grid_tomorrow,
                self.higher_level_buy_price,
                self.higher_level_sell_price,
                prev_energy,)
            )#.astype(np.float32)

    def step(self, action):
        """
        purpose: a single step for one lower level aggregator.

        Args:
            Action: a lower level aggregator buy and sell price 
        
        Returns:
            Observation: 
                generation_tomorrow;
                grid buy, sell; 
                higher agg buy, sell; 
                prev_energy
            Reward, Done, Info: you know the deal
        """ 

        self.action = action
        self.curr_iter += 1
        self.total_iter += 1

        done = {
            self.curr_iter > 0
        }

        buyprice, sellprice = self._price_from_action(action)
            # self.price = price

        obs = self._get_observation()

        grid_buy_price = obs[24:48]
        grid_sell_price = obs[48:72]
        higher_aggregator_buy_price = obs[72:96]
        higher_aggregator_sell_price = obs[96:120]

        check = True

        if check:
            buyprice_grid = self.buyprices_grid[self.day]
            sellprice_grid = self.sellprices_grid[self.day]
            # pdb.set_trace()
            if not sum(grid_buy_price == buyprice_grid) == len(grid_buy_price):
                raise AssertionError(f"grid_buy_price: {grid_buy_price} not equal \
                    to buyprice_grid{buyprice_grid}" )
            assert sum(grid_sell_price == sellprice_grid) == len(grid_sell_price)

        optimal_prosumer_buyprice = np.minimum(
            grid_buy_price, np.minimum(
                higher_aggregator_buy_price, 
                buyprice)
            )

        optimal_prosumer_sellprice = np.maximum(
            grid_sell_price, np.maximum(
                higher_aggregator_sell_price, 
                sellprice)
            )

        energy_consumptions = self._simulate_prosumers_twoprices(
            day = self.day, 
            buyprice = optimal_prosumer_buyprice, 
            sellprice = optimal_prosumer_sellprice)

        self.prev_energy = energy_consumptions["Total"]
        
        optimal_aggregator_buyprice = np.minimum(
            grid_buy_price,
            higher_aggregator_buy_price
        )
        optimal_aggregator_sellprice = np.maximum(
            grid_sell_price,
            higher_aggregator_sell_price
        )
    
        reward = self._get_reward_twoprices(
            optimal_aggregator_buyprice, 
            optimal_aggregator_sellprice, 
            buyprice, 
            sellprice, 
        energy_consumptions)

        next_observation = self._get_observation()
        
        self.iteration += 1

        return next_observation, reward, done, {}


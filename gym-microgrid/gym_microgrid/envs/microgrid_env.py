import gym
from gym import spaces

import numpy as np
import pandas as pd 
import random

from gym_microgrid.envs.utils import price_signal
from gym_microgrid.envs.agents import *
from gym_microgrid.envs.reward import Reward
from gym_socialgame.envs.buffers import GaussianBuffer
from multiprocessing import Pool, pool
from copy import deepcopy

import pathlib

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import wandb
CSV_PATH = pathlib.Path(__file__).parent / "building_data.csv"
CSV_PATH = CSV_PATH.resolve()

DAY_LENGTH = 24

def pool_fn(x):
    (prosumer_name, prosumer), day, buyprice, sellprice = x
    energy, batt_times, batt_cap = prosumer.get_response_twoprices(
        day, buyprice, sellprice)
    return prosumer_name, energy, batt_times, batt_cap  # what you'd think x + *y does

class MicrogridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        action_space_string = "continuous",
        number_of_participants = 10,
        one_day = 0,
        energy_in_state = False,
        reward_function = "market_solving",
        complex_batt_pv_scenario=1,
        exp_name = None,
        smirl_weight=None,
        num_mg_optim_steps=10000,
        num_mg_workers=10,
        max_episode_steps=365,
        starting_day=None, 
        no_rl_control=False,
        no_external_grid=False,
        **kwargs
        ):
        """
        MicrogridEnv for an agent determining incentives in a social game.

        Note: One-step trajectory (i.e. agent submits a 24-dim vector containing transactive price for each hour of each day.
            Then, environment advances one-day and agent is told that the episode has finished.)

        Args:
            action_space_string: (str) either "continuous" or "multidiscrete"
            number_of_participants: (int) denoting the number of players in the social game (must be > 0 and < 20)
            one_day: (int) in range [-1,365] denoting which fixed day to train on .
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,365] = Day of the Year
            energy_in_state: (bool) denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: (bool) denoting whether (or not) to append yesterday's price signal to the state
            max_episode_steps: (int) Number of times .step() is called before done is returned.
            starting_day: (Optional[int]) Starting day, or None for random.

        """
        super(MicrogridEnv, self).__init__()

        #Verify that inputs are valid
        self.check_valid_init_inputs(
            action_space_string,
            number_of_participants,
            one_day,
            energy_in_state,
        )
        #Assigning Instance Variables
        self.action_space_string = action_space_string
        self.number_of_participants = number_of_participants
        self.num_optim_steps = num_mg_optim_steps
        self.energy_in_state = energy_in_state
        self.reward_function = reward_function
        self.complex_batt_pv_scenario = complex_batt_pv_scenario
        self.starting_day = starting_day
        self.no_rl_control=no_rl_control
        self.no_external_grid = no_external_grid

        self.smirl_weight = smirl_weight
        self.use_smirl = smirl_weight > 0 if smirl_weight else False

        self.last_smirl_reward = None
        self.last_energy_reward = None

        #Create Observation Space (aka State Space)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(72,), dtype=np.float32)

        self.buyprices_grid, self.sellprices_grid = self._get_prices()
        # self.prices = self.buyprices_grid # Initialise to buyprices_grid
        self.generation = self._get_generation()

        #Create Action Space
        self.action_length = 2 * DAY_LENGTH 
        self.action_subspace = 3
        self.action_space = spaces.Box(low = -1, high = 1, shape = (2 * DAY_LENGTH, ), dtype = np.float32)

        #Create Prosumers
        self.prosumer_dict = self._create_agents()

        if self.use_smirl:
            self.buffer = GaussianBuffer(self.action_length)
        self.max_episode_steps = max_episode_steps

        self.num_workers = num_mg_workers
        print("Number of workers in pool: " + str(self.num_workers))
        self.pool=Pool(processes=self.num_workers)

        self.last_metrics = {}
        
        self.reset()

        print("\n Microgrid Environment Initialized! Have Fun! \n")
    
    def set_starting_day(self, day):
        """Sets the starting day; use this before calling
            reset to synchronize a day for all the microgrids"""
        self.starting_day = day
    
    def reset(self):
        """ Resets the environment on the current day """
        #Cur_iter counts length of trajectory for current step (i.e. cur_iter = i^th hour in a 10-hour trajectory)
        #For our case cur_iter just flips between 0-1 (b/c 1-step trajectory)
        self.timestep = 0
        if self.starting_day is not None:
            self.day = self.starting_day
        else:
            self.day = np.random.randint(0, 365)
        self.action = np.repeat(np.nan, self.action_length)
        #TODO: Check initialization of prev_energy
        self.prev_energy = np.zeros(DAY_LENGTH)
        self.last_metrics["total_batt_discharged_capacity"] = 0
        self.last_metrics["total_discharged_time"] = 0
        self.last_metrics["money_from_prosumers"] = 0
        self.last_metrics["money_to_utility"] = 0
        self.last_metrics["daily_violations"] = 0
        self.last_metrics["max_proportion"] = 0
        return self._get_observation()

    def _create_agents(self):
        """
        Purpose: Create the prosumers in the local energy market. 
        We create a market with n players, where n = self.number_of_participants

       Args:
            None

        Returns:
              prosumer_dict: Dictionary of prosumers

        """
        prosumer_dict = {}

        # Manually set battery numbers and PV sizes

        ## large and constant batt and PV
        if self.complex_batt_pv_scenario == 1:
            battery_nums = [50]*self.number_of_participants
            pvsizes = [100]*self.number_of_participants

        ## small PV sizes
        elif self.complwex_batt_pv_scenario == 2: 
            pvsizes = [ 0, 10, 100, 10, 0, 0, 0, 55, 10, 10 ]
            battery_nums = [ 0, 0, 50, 30, 50, 0, 0, 10, 40, 50 ]

        ## medium PV sizes and different
        elif self.complex_batt_pv_scenario == 3:
            pvsizes = [ 70, 110, 400, 70, 30, 0, 0, 55, 10, 20 ]
            battery_nums = [ 0, 0, 150, 30, 50, 0, 0, 100, 40, 150]

        ## no batteries 
        elif self.complex_batt_pv_scenario == 4:
            pvsizes = [ 70, 110, 400, 70, 30, 0, 0, 55, 10, 20 ]
            battery_nums = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        
        # no solar 
        elif self.complex_batt_pv_scenario == 5:
            pvsizes = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            battery_nums = [ 0, 0, 150, 30, 50, 0, 0, 100, 40, 150 ]
        
        # nothing at all 
        elif self.complex_batt_pv_scenario == 6:
            pvsizes = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            battery_nums = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

        # wrong number 
        else:
            print("you've inputted an incorrect scenario")
            raise AssertionError

        pvsizes = pvsizes[:self.number_of_participants]
        battery_nums = battery_nums[:self.number_of_participants]

        self.pv_sizes = pvsizes
        self.battery_sizes = battery_nums

        # Get energy from building_data.csv file,  each office building has readings in kWh. Interpolate to fill missing values
        df = pd.read_csv(CSV_PATH).interpolate().fillna(0)
        building_names = df.columns[5:] # Skip first few columns 
        for i in range(self.number_of_participants):
            name = building_names[i]
            prosumer = Prosumer(name, np.squeeze(df[[name]].values), .001*np.squeeze(df[['PV (W)']].values), battery_num = battery_nums[i], pv_size = pvsizes[i], num_optim_steps=self.num_optim_steps)
            prosumer_dict[name] = prosumer
        return prosumer_dict
    
    @staticmethod
    def _get_generation():
        """
        Purpose: Get solar energy predictions for the entire year 

        Args:
            None

        Returns: Array containing solar generation predictions, where array[day_number] = renewable prediction for day_number 
        """
        yearlonggeneration = []
        # Read renewable generation from CSV file. Index starts at 5 am on Jan 1, make appropriate adjustments. For year 2012: it is a leap year
        # generation = pd.read_csv('/Users/utkarshapets/Documents/Research/Optimisation attempts/building_data.csv')[['PV (W)']]
        generation = np.squeeze(pd.read_csv(CSV_PATH)[['PV (W)']].values)
        for day in range(0, 365):
            yearlonggeneration.append(
                generation[day*DAY_LENGTH+19 : day*DAY_LENGTH+19+24]
            )
        return np.array(yearlonggeneration)

    @staticmethod
    def _get_prices():
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
        price = np.squeeze(pd.read_csv(CSV_PATH)[['Price( $ per kWh)']].values)
        for day in range(365):
            buyprice = price[day*DAY_LENGTH+19 : day*DAY_LENGTH+19+24]
            sellprice = 0.6*buyprice
            buy_prices.append(buyprice)
            sell_prices.append(sellprice)
        return buy_prices, sell_prices

    def _price_from_action(self, action):
        """
        Purpose: Convert agent actions that lie in [-1,1] into transactive price (conversion is for multidiscrete setting)

        Args:
            Action: 24-dim vector corresponding to action for each hour

        Returns: Price: 24-dim vector of transactive prices
        """
        
        # Continuous space is symmetric [-1,1], we map to -> [sellprice_grid,buyprice_grid] 
        buyprice_grid = self.buyprices_grid[self.day]
        sellprice_grid = self.sellprices_grid[self.day]
        
        # -1 -> sellprice. 1 -> buyprice
        midpoint_price = (buyprice_grid + sellprice_grid)/2
        diff_grid = buyprice_grid - sellprice_grid
        scaled_diffs_bp = np.multiply(action[0:24], diff_grid)/2 # Scale to fit difference at each hour
        scaled_diffs_sp = np.multiply(action[24:], diff_grid)/2 # Scale to fit difference at each hour
        buyprice = scaled_diffs_bp + midpoint_price
        sellprice = scaled_diffs_sp + midpoint_price
        return buyprice, sellprice

    def _simulate_prosumers_twoprices(self):
        """
        Purpose: Gets energy consumption from players given action from agent
                 Price: transactive price set in day-ahead manner

        Returns:
            Energy_consumption: Dictionary containing the energy usage by prosumer. 
                Key 'Total': aggregate net energy consumption
        """
        
        energy_consumptions = {}
        batt_discharge_times = {}
        batt_discharge_capacities = {}
        total_consumption = np.zeros(24)
        day_list = [self.day for _ in range(len(self.prosumer_dict))]
        
        if self.no_external_grid:
            buyprice = self.buyprice
            sellprice= self.sellprice
        else:
            buyprice = np.minimum(self.buyprice, self.buyprices_grid[self.day])
            sellprice = np.maximum(self.sellprice, self.sellprices_grid[self.day])
        if self.num_workers > 1:
            buyprice_list = [buyprice for _ in range(len(self.prosumer_dict))]
            sellprice_list = [sellprice for _ in range(len(self.prosumer_dict))]
            prosumer_names, prosumer_demands, batt_discharges, batt_capacities = \
                    zip(*self.pool.map(
                        pool_fn, 
                            zip(
                                self.prosumer_dict.items(), 
                                day_list, 
                                buyprice_list, 
                                sellprice_list)
                                ))
            for prosumer_name, prosumer_demand, batt_discharge, batt_capacity in zip(
                    prosumer_names, prosumer_demands, batt_discharges, batt_capacities):
                prosumer = self.prosumer_dict[prosumer_name]
                batt_discharge_capacities[prosumer_name] = batt_capacity
                batt_discharge_times[prosumer_name] = batt_discharge
                energy_consumptions[prosumer_name] = prosumer_demand
                total_consumption += prosumer_demand
        else:
            for prosumer_name in self.prosumer_dict:
                #Get players response to agent's actions
                prosumer = self.prosumer_dict[prosumer_name]
                prosumer_demand, batt_discharge, batt_capacity = (
                    prosumer.get_response_twoprices(self.day, buyprice, sellprice))
                
                #Calculate energy consumption by prosumer and in total (entire aggregation)
                energy_consumptions[prosumer_name] = prosumer_demand
                batt_discharge_capacities[prosumer_name] = batt_capacity
                batt_discharge_times[prosumer_name] = batt_discharge
                total_consumption += prosumer_demand
        energy_consumptions["Total"] = total_consumption 
        return energy_consumptions, batt_discharge_capacities, batt_discharge_times

    def _get_reward_twoprices(self):
        """
        Purpose: Compute reward given grid prices, transactive price set ahead of time, and energy consumption of the participants

        Returns:
            Reward for RL agent (- |net money flow|): in order to get close to market equilibrium
        """
        buyprice_grid = self.buyprices_grid[self.day]
        sellprice_grid = self.sellprices_grid[self.day]
        total_consumption = self.energy_consumptions['Total']

        test_buy_from_grid = self.buyprice < buyprice_grid # Bool vector containing when prosumers buy from microgrid
        test_sell_to_grid = self.sellprice > sellprice_grid # Bool vector containing when prosumers sell to microgrid
        if np.all(self.buyprice == buyprice_grid):
            test_buy_from_grid = np.repeat(.5, buyprice_grid.shape)
        if np.all(self.sellprice == sellprice_grid):
            test_sell_to_grid = np.repeat(.5, sellprice_grid.shape)

        money_to_utility = (np.dot(np.maximum(0, total_consumption * test_buy_from_grid), buyprice_grid) - 
            np.dot(np.minimum(0, total_consumption * test_sell_to_grid), sellprice_grid))

        money_from_prosumers = 0
        for prosumerName in self.energy_consumptions:
            if prosumerName != "Total":
                money_from_prosumers += (
                    np.dot(np.maximum(0, self.energy_consumptions[prosumerName]) * test_buy_from_grid, self.buyprice) - 
                    np.dot(np.minimum(0, self.energy_consumptions[prosumerName]) * test_sell_to_grid, self.sellprice))

        self.last_metrics["money_from_prosumers"] = money_from_prosumers
        self.last_metrics["money_to_utility"] = money_to_utility

        # self.money_from_prosumers.append(money_from_prosumers)
        # self.money_to_utility.append(money_to_utility)

        total_reward = None
        if self.reward_function == "market_solving":
            total_reward = - abs(money_from_prosumers - money_to_utility)
        elif self.reward_function =="profit_maximizing":
            total_reward = money_from_prosumers - money_to_utility

        return total_reward

    
    def _count_voltage_constraints(self):
        """
        Purpose: Counts the number of times energy expenditure exceeds the power rating 
        of the transformer. v0 assumes fixed charging throughout the hour and assumes
        that transformers are sized up to ~1.25 of the max power observed in baseline
        energy consumption per building
        
        Returns:
            violations: count of voltage constraint violations throughout the day
            proportion_matched: continuous proportion of power rating per hour 
        """
        buildings = list(self.prosumer_dict.keys())
        transformer_ratings = [50, 65, 600, 150, 200, 150, 150, 450, 175, 600]
        transformer_dict = {
            buildings[i]: transformer_ratings[i] for i in list(range(len(buildings)))}

        violations = {}
        proportion_matched = {}
        for prosumer_name in buildings:
            violations[prosumer_name] = np.sum(
                np.array(self.energy_consumptions[prosumer_name]) > transformer_dict[prosumer_name])
            proportion_matched[prosumer_name] = (
                np.array(self.energy_consumptions[prosumer_name]) / transformer_dict[prosumer_name]
            )
        
        return violations, proportion_matched

    def step(self, action):
        """
        Purpose: Takes a step in the environment

        Args:
            action: 24 dim vector over [-1, 1]

        Returns:
            next_obs: State for the next day
            reward: Reward for today's action
            done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            info: Other info (primarily for gym env based library compatibility)

        Exceptions:
            raises AssertionError if action is not in the action space
        """

        action = np.clip(action, -1, 1)  # TODO this should never be used because action_space = Box(-1, 1)?
        
        self.action = action
        self.buyprice, self.sellprice = self._price_from_action(action)
        if self.no_rl_control:
            self.buyprice, self.sellprice = self.buyprices_grid[self.day], self.sellprices_grid[self.day]
            
        self.energy_consumptions, batt_discharged_capacities, batt_discharged_times_n = (
                self._simulate_prosumers_twoprices())
        reward = self._get_reward_twoprices()
        self.prev_energy = self.energy_consumptions["Total"]
        num_violations, proportions = self._count_voltage_constraints()
        self.last_metrics["daily_violations"] = sum(num_violations.values()) / len(num_violations)
        self.last_metrics["max_proportion"] = max([max(p) for p in proportions.values()])
        self.last_metrics["total_batt_discharged_capacity"] = np.sum(list(batt_discharged_capacities.values()))
        self.last_metrics["total_discharged_time"] = np.sum(list(batt_discharged_times_n.values()))
        # self.daily_violations.append(sum(num_violations.values()) / len(num_violations))
        # self.max_proportion.append(max([max(p) for p in proportions.values()]))
                  
        # self.total_batt_discharged_capacities.append(np.sum(list(batt_discharged_capacities.values())))
        # self.total_discharged_times.append(np.sum(list(batt_discharged_times_n.values())))

        if self.use_smirl:
            raise NotImplementedError
            # self.buffer.add(next_obs)
        
        # if self.timestep == self.max_episode_steps - 1:
        #     wandb.log({"Money_From_Prosumers": wandb.Histogram(self.money_from_prosumers),
        #                "Money To Utility": wandb.Histogram(self.money_to_utility),
        #                "Total Batt Discharged Capacities": wandb.Histogram(self.total_batt_discharged_capacities),
        #                "Total Discharged Times": wandb.Histogram(self.total_discharged_times),
        #                "Mean Daily Violations Per Building": wandb.Histogram(self.daily_violations),
        #                "Max Transformer Capacity Proportion": wandb.Histogram(self.max_proportion)}, commit=False)

        # violations, proportions_matched = self._count_voltage_constraints()
        # voltage_risks = max(max(v) for v in proportions_matched.values()) ### TODO SAM: is this a good measure?

        # if self.ancillary_logger:
        #     self.ancillary_logger.log({"Money_From_Prosumers":  self.money_from_prosumers[-1],
        #                 "Money To Utility":  self.money_to_utility[-1],
        #                 "Total Batt Discharged Capacities":  self.total_batt_discharged_capacities[-1],
        #                 "Total Discharged Times":  self.total_discharged_times[-1],
        #                 "Mean Daily Violations Per Building": (self.daily_violations[-1]),
        #                 "Max Transformer Capacity Proportion": (self.max_proportion[-1]),
        #                 "Voltage Risks": voltage_risks})

        self.timestep += 1
        self.day = (self.day + 1) % 365
        next_obs = self._get_observation()
        done = self.timestep == self.max_episode_steps  # TODO One should use TimeLimit wrapper, not this...
        return next_obs, reward, done, {}
    

    def _get_observation(self):
        """Get today's observation."""
        generation_today = self.generation[self.day] 
        buyprice_grid_today = self.buyprices_grid[self.day]

        noise = np.random.normal(loc = 0, scale = 50, size = 24) ## TODO: get rid of this if not doing well
        generation_today_nonzero = (generation_today > abs(noise)) # when is generation non zero?
        generation_today += generation_today_nonzero * noise # Add in Gaussian noise when gen in non zero

        return np.concatenate(
            (self.prev_energy, generation_today, buyprice_grid_today)
            ).astype(np.float32)
        

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    @staticmethod
    def check_valid_init_inputs(action_space_string: str, number_of_participants = 10,
                one_day = False, energy_in_state = False):

        """
        Purpose: Verify that all initialization variables are valid

        Args (from initialization):
            action_space_string: String either "continuous" or "discrete" ; Denotes the type of action space
            number_of_participants: Int denoting the number of players in the social game (must be > 0 and < 20)
            one_day: Boolean denoting whether (or not) the environment is FIXED on ONE price signal
            energy_in_state: Boolean denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: Boolean denoting whether (or not) to append yesterday's price signal to the state

        Exceptions:
            Raises AssertionError if action_space_string is not a String or if it is not either "continuous", or "multidiscrete"
            Raises AssertionError if number_of_participants is not an integer, is less than 1,  or greater than 20 (upper bound set arbitrarily for comp. purposes).
            Raises AssertionError if any of {one_day, energy_in_state, yesterday_in_state} is not a Boolean
        """

        #Checking that action_space_string is valid
        assert isinstance(action_space_string, str), "action_space_str is not of type String. Instead got type {}".format(type(action_space_string))
        action_space_string = action_space_string.lower()
        assert action_space_string in ["continuous", "multidiscrete", "continuous_normalized"], "action_space_str is not continuous or discrete. Instead got value {}".format(action_space_string)

        #Checking that number_of_participants is valid
        assert isinstance(number_of_participants, int), "Variable number_of_participants is not of type Integer. Instead got type {}".format(type(number_of_participants))
        assert number_of_participants > 0, "Variable number_of_participants should be atleast 1, got number_of_participants = {}".format(number_of_participants)
        assert number_of_participants <= 20, "Variable number_of_participants should not be greater than 20, got number_of_participants = {}".format(number_of_participants)

        #Checking that one_day is valid
        assert isinstance(one_day, int), "Variable one_day is not of type Int. Instead got type {}".format(type(one_day))
        assert 366 > one_day and one_day > -2, "Variable one_day out of range [-1,365]. Got one_day = {}".format(one_day)

        #Checking that energy_in_state is valid
        assert isinstance(energy_in_state, bool), "Variable one_day is not of type Boolean. Instead got type {}".format(type(energy_in_state))

        print("all inputs valid")


class MicrogridEnvRLLib(MicrogridEnv):
    """ 
    Child Class of MicrogridEnv to support RLLib. 
    two_price_state and complex_batt_pv are specific params for the MicrogridEnv
    and differs from SocialGame. 
    """
    def __init__(self, env_config):
        super(MicrogridEnvRLLib, self).__init__(**env_config)
        print("Initialized RLLib child class for MicrogridEnv.")

class CounterfactualMicrogridEnvRLLib(MicrogridEnvRLLib, MultiAgentEnv):
    """A modification of Microgrid env to include a counterfactual grid 

    This script creates a regular grid and a shadow grid, in which the agent
    will try to do as poorly as possible. This may help the agent understand
    more about the space of actions. 
    """
    def __init__(self, env_config):
        super(CounterfactualMicrogridEnvRLLib, self).__init__(env_config)
        self.two_price_state = True
        self.prev_energy = {
            "real":np.zeros(self.day_length),
            "shadow":np.zeros(self.day_length)
        }
        self.last_energy_cost = {
            "real": 1,
            "shadow": 1
        }
        self.last_energy_reward = {
            "real": np.repeat(1, 48),
            "shadow": np.repeat(1, 48),
        }
        self.action_dict = {
            "real": np.repeat(1, 48),
            "shadow": np.repeat(1, 48),
        }
        


    def _get_reward_twoprices(
            self, 
            buyprice_grid, 
            sellprice_grid, 
            transactive_buyprice, 
            transactive_sellprice, 
            energy_consumptions
        ):
        """
        Purpose: Compute positive reward for the real agent and negative reward for the shadow agent

        Args:
            buyprice_grid: price at which energy is bought from the utility (24 dim vector)
            sellprice_grid: price at which energy is sold to the utility by the RL agent (24 dim vector)
            transactive_buyprice: price set by RL agents for local market in day ahead manner (dict of 24 dim vectors)
            transactive_sellprice: price set by RL agents for local market in day ahead manner (dict of 24 dim vectors)
            energy_consumptions: Dictionary of dicts containing energy usage by each prosumer, as well as the total

        Returns:
            Reward for RL agent (- |net money flow|): in order to get close to market equilibrium
        """

        total_consumption_real = energy_consumptions["real"]['Total']
        total_consumption_shadow = energy_consumptions["shadow"]["Total"]

        money_to_utility_real = (
            np.dot(np.maximum(0, total_consumption_real), buyprice_grid) + 
            np.dot(np.minimum(0, total_consumption_real), sellprice_grid)
        )
        money_to_utility_shadow = (
            np.dot(np.maximum(0, total_consumption_shadow), buyprice_grid) + 
            np.dot(np.minimum(0, total_consumption_shadow), sellprice_grid)
        )

        money_from_prosumers_real = 0
        money_from_prosumers_shadow = 0


        for prosumerName in energy_consumptions["real"]:
            money_from_prosumers_real += (
                np.dot(
                    np.maximum(0, energy_consumptions["real"][prosumerName]), 
                    transactive_buyprice["real"]) + 
                np.dot(
                    np.minimum(0, energy_consumptions["real"][prosumerName]), 
                    transactive_sellprice["real"])
            )

        for prosumerName in energy_consumptions["shadow"]:
            money_from_prosumers_shadow += (
                np.dot(
                    np.maximum(0, energy_consumptions["shadow"][prosumerName]), 
                    transactive_buyprice["shadow"]) + 
                np.dot(
                    np.minimum(0, energy_consumptions["shadow"][prosumerName]), 
                    transactive_sellprice["shadow"])
            )

        # IPython.embed()

        if self.reward_function == "market_solving":
            total_reward_real = - abs(money_from_prosumers_real - money_to_utility_real)
            total_reward_shadow = - abs(money_from_prosumers_shadow - money_to_utility_shadow)

        elif self.reward_function == "profit_maximizing":
            total_reward_real = money_from_prosumers_real - money_to_utility_real
            total_reward_shadow = money_from_prosumers_shadow - money_to_utility_shadow
        
        else:
            raise ValueError("Your reward type is not supported. Choose profit maximizing")

        self.last_energy_reward = self.last_energy_cost = {
            "real": total_reward_real,
            "shadow": total_reward_shadow
        }


        return {"real":total_reward_real,
            "shadow":-total_reward_shadow}

    def _get_observation(self):
    
        print("getting obs")
        prev_energy = self.prev_energy
        generation_tomorrow = self.generation[(self.day + 1)%365] 
        buyprice_grid_tomorrow = self.buyprices_grid[(self.day + 1)%365] 

        noise = np.random.normal(loc = 0, scale = 50, size = 24) ## TODO: get rid of this if not doing well
        generation_tomorrow_nonzero = (generation_tomorrow > abs(noise)) # when is generation non zero?
        generation_tomorrow += generation_tomorrow_nonzero* noise # Add in Gaussian noise when gen in non zero

        return {
            "real": np.concatenate(
                (prev_energy["real"], generation_tomorrow, buyprice_grid_tomorrow)
            ),
            "shadow": np.concatenate(
                (prev_energy["shadow"], generation_tomorrow, buyprice_grid_tomorrow)
            )
        }
     

    def step(self, action_dict):
        """
        Purpose: Takes a step in the environment for each agent 

        Args:
            Action: 24 dim vector in [-1, 1]

        Returns:
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        Exceptions:
            raises AssertionError if action is not in the action space
        """

        self.action_dict = action_dict

        if not self.action_space.contains(action_dict["real"]):
            action_real = np.asarray(action_dict["real"])
            action_shadow = np.asarray(action_dict["shadow"])
            if self.action_space_string == 'continuous':
                action_real = np.clip(action_real, -1, 1)
                action_shadow = np.clip(action_shadow, -1, 1)
                # TODO: ask Lucas about this
            else:
                print("wrong action_space_string")
                raise AssertionError
        else:
            action_real = np.asarray(action_dict["real"])
            action_shadow = np.asarray(action_dict["shadow"])

        # prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365 
        self.curr_iter += 1
        self.total_iter += 1

        done = {
            "real": self.curr_iter > 0,
            "shadow": self.curr_iter > 0,
            "__all__": self.curr_iter > 0
        }

        # IPython.embed()

        if not self.two_price_state:
            raise ValueError("one price state not supported in the MACRL")
            # price = self._price_from_action(action)
            # self.price = price

            # energy_consumptions = self._simulate_humans(day = self.day, price = price)
            # self.prev_energy = energy_consumptions["Total"]

            # observation = self._get_observation()
        
            # buyprice_grid = self.buyprices_grid[self.day]
            # sellprice_grid = self.sellprices_grid[self.day]
            # reward = self._get_reward(buyprice_grid, sellprice_grid, price, energy_consumptions)

            # self.iteration += 1

        else: 
            buyprice, sellprice = {}, {}
            buyprice["real"], sellprice["real"] = self._price_from_action(action_real)
            buyprice["shadow"], sellprice["shadow"] = self._price_from_action(action_shadow)
            # self.price = price

            energy_consumptions = {}
            energy_consumptions["real"] = self._simulate_prosumers_twoprices(
                day = self.day, 
                buyprice = buyprice["real"], 
                sellprice = sellprice["real"])
            
            energy_consumptions["shadow"] = self._simulate_prosumers_twoprices(
                day = self.day, 
                buyprice = buyprice["shadow"], 
                sellprice = sellprice["shadow"])

            self.prev_energy = {
                "real":energy_consumptions["real"]["Total"],
                "shadow":energy_consumptions["shadow"]["Total"]
            }

            observation = self._get_observation() # this already turns back a dict

        
            buyprice_grid = self.buyprices_grid[self.day]
            sellprice_grid = self.sellprices_grid[self.day]
            reward = self._get_reward_twoprices(
                buyprice_grid, 
                sellprice_grid, 
                buyprice, 
                sellprice, 
                energy_consumptions
            )
            
            self.iteration += 1

        self.store_sample_user(energy_consumptions)

        if self.use_smirl:
            self.buffer.add(observation)

        info = {"real":{}, "shadow":{}}
        return observation, reward, done, info

    def store_sample_user(self, energy_consumptions):
        """ Stores the sample user reactions per day. 
        
        Args:
            energy_consumptions: output from simulate_humans
        """

        self.sample_user_response["sample_user"] = (
            "scenario_" + 
            str(self.complex_batt_pv_scenario) + 
            "_user_" + 
            str(self.sample_user))


        self.sample_user_response["pv_size"] = self.pv_sizes[self.sample_user]
        self.sample_user_response["battery_size"] = self.battery_sizes[self.sample_user]

        # get the building

        prosumer = list(self.prosumer_dict.keys())[self.sample_user]

        for key, val in energy_consumptions.items():
            for i, k in enumerate(val[prosumer].flatten()):
                self.sample_user_response[key]["prosumer_response_hour_" + str(i)] = k

        if not self.iteration % 10:
            self.sample_user = np.random.choice(10, 1)[0]

        self.energy_consumption_length = len(self.sample_user_response["real"])

        return

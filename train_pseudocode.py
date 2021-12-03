####
# Example train loop
###


"""
num_lower_level_agents = 5
upper_level_agent = feudal_env.FeudalSocialGameUpper2HourAgent(num_lower_agents=num_lower_level_agents)
lower_level_agents = [{i: feudal_env.FeudalSocialGameLower3HourAgent(num=i)} for i in range(num_lower_level_agents)]

def train_one_step():

    lower_level_action = []
    concatenated_action = []
    observations = []
    rewards = []
    done = []
    info = []

    observation = upper_level_agent._get_observation()

    upper_level_command = upper_level_agent.get_upper_level_agent_action(observation)

    for i in range(num_lower_level_agents):
        lower_level_agents[i].set_upper_level_command(upper_level_command[(i*2) : (i*2 + 1)])
        action[i] = lower_level_agents[i].get_action()
        concatenated_action += action

    upper_level_agent.set_yesterdays_points(concatenated_action)
    upper_level_observation, upper_level_reward, upper_level_done, upper_level_info = (
        upper_level_agent.step(upper_level_command)
        )

    total_demand = upper_level_agent.prev_energy
    
    for i in range(num_lower_level_agents):
        lower_level_agents.set_total_demand_attribute(total_demand)
        lower_level_step_results = lower_level_agents[i].step(action[i])
        observations[i] = lower_level_step_results[0]
        rewards[i] = lower_level_step_results[1]
        done[i] = lower_level_step_results[2]
        info[i] = lower_level_step_results[3]

"""

import argparse
import gym
import numpy as np
import os
import wandb
import math
import utils
import json
from custom_callbacks import CustomCallbacks, MultiAgentCallbacks
from copy import deepcopy

import ray
import ray.rllib.agents.ppo as ray_ppo
from ray.rllib.examples.models.centralized_critic_models import \
    CentralizedCriticModel, TorchCentralizedCriticModel
from ray.rllib.models import ModelCatalog
from pfl_hypernet import PFL_Hypernet
from ccppo import CCPPOTorchPolicy, CCTrainer

from gym_socialgame.envs.socialgame_env import (SocialGameEnvRLLib)
from gym_microgrid.envs.microgrid_env import (MicrogridEnvRLLib)
from gym_microgrid.envs.multiagent_env import (MultiAgentSocialGameEnv, MultiAgentMicrogridEnv)

import torch
import torch.optim as optim
from collections import OrderedDict


hnet = None
hnet_optimizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

def get_agent(args):
    global hnet, hnet_optimizer, device
    """
    Purpose: Import the algorithm and policy to create an agent. 
    Returns: Agent
    Exceptions: Malformed args to create an agent. 
    """
    ### Setup for MultiAgent ###
    #Create a dummy environment to get action and observation space for our settings
    dummy_env = environments[args.gym_env](vars(args))
    # TODO: HACK
    # dummy_env = environments["socialgame"](vars(args))
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    config = {}
    if args.num_gpus > 0:
        device = "cuda"
    if args.gym_env in ["socialgame_multi", "microgrid_multi"]:
        if args.use_hnet:
            hnet = PFL_Hypernet(n_nodes = dummy_env.n_nodes,
                        embedding_dim = args.hnet_embedding_dim, 
                        num_layers = args.hnet_num_layers,
                        num_hidden = args.hnet_num_hidden,
                        out_params_path = args.hnet_out_params_path,
                        lr = args.hnet_lr,
                        device = device)
            hnet_optimizer = optim.Adam(hnet.parameters(), lr=args.hnet_lr)
        if args.algo=="ppo":
            #pass
            config["multiagent"] = {
                "policies": {str(i): (ray_ppo.PPOTorchPolicy, obs_space, act_space, {"number_of_agents": dummy_env.n_nodes}) for i, scenario in enumerate(range(dummy_env.n_nodes))},
                "policy_mapping_fn": lambda agent_id, episode=None: agent_id
            }
        elif args.algo=="single_ppo":
            print("USING SINGLE PPO MODEL")
            config["multiagent"] = {
                "policies": {'0': (ray_ppo.PPOTorchPolicy, obs_space, act_space, {"number_of_agents": dummy_env.n_nodes}) },
                "policy_mapping_fn": lambda agent_id, episode=None: '0'
            }
        elif args.algo=="ccppo":
            config["multiagent"] = {
                "policies": {str(i): (CCPPOTorchPolicy, obs_space, act_space, {"number_of_agents": dummy_env.n_nodes}) for i, scenario in enumerate(range(dummy_env.n_nodes))},
                "policy_mapping_fn": lambda agent_id, episode=None: agent_id
            }
        else:
            raise NotImplementedError

    #### Algorithm: PPO ####
    if args.algo == "ppo" or "ccppo" or "single_ppo":
        # Modify the default configs for PPO
        default_config = ray_ppo.DEFAULT_CONFIG.copy()
        #merge config with default config (with ours overwriting in case of clashes)
        config = {**default_config, **config}
        config["framework"] = "torch"
        config["train_batch_size"] = 256
        config["sgd_minibatch_size"] = 16
        config["lr"] = 3e-4
        config["clip_param"] = 0.3
        config["num_gpus"] =  args.num_gpus # this may throw an error
        config["num_workers"] = args.num_workers
        config["env_config"] = vars(args)
        config["env"] = environments[args.gym_env]
        # obs_dim = np.sum([args.energy_in_state, args.price_in_state])
        obs_dim = math.prod(obs_space.shape)
            
        out_path = os.path.join(args.log_path, "bulk_data.h5")
        if args.gym_env in ["socialgame_multi", "microgrid_multi"]:
            callbacks = MultiAgentCallbacks(log_path=out_path, save_interval=args.bulk_log_interval, 
                                            obs_dim=obs_dim, num_agents=dummy_env.n_nodes)
        else:
            callbacks = CustomCallbacks(log_path=out_path, save_interval=args.bulk_log_interval, obs_dim=obs_dim)

        config["callbacks"] = lambda: callbacks
        logger_creator = utils.custom_logger_creator(args.log_path)

        callbacks.save()
        print("Saved first callback")

        if args.wandb:
            wandb.save(out_path)
        if args.algo == "ppo" or "single_ppo":
            return ray_ppo.PPOTrainer(
                config = config, 
                env = environments[args.gym_env],
                logger_creator = logger_creator
            )
        else:
            print("RUNNING CCPPO")
            ModelCatalog.register_custom_model(
                "cc_model", TorchCentralizedCriticModel
                if config["framework"] == "torch" else CentralizedCriticModel)
            config["model"] = {"custom_model": "cc_model",}
            return CCTrainer(
                config = config, 
                env = environments[args.gym_env],
                logger_creator = logger_creator
            )

    # Add more algorithms here. 

def pfl_hnet_update(agent, result, args, old_weights):
    # Get list of weights of each worker, including remote replicas
    #curr_weights = trainer.workers.foreach_worker(lambda ev: ev.get_policy().get_weights())
    curr_weights = agent.get_weights()
    # set gradients to 0 to start w
    hnet_optimizer.zero_grad()
    # update hnet weights
    num_agents = len(curr_weights)
    for agent_id in curr_weights.keys():
        # calculating delta theta
        
        delta_theta = OrderedDict({k: old_weights[agent_id][k].to(device) - torch.tensor(curr_weights[agent_id][k], device=device) for k in curr_weights[agent_id].keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            [v.to(device) for v in old_weights[agent_id].values()], hnet.parameters(), grad_outputs=list(delta_theta.values())
        )
        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            # normalize so we end up with a grad that is the mean of all the updates from each agent
            p.grad = g / num_agents 
    torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
    hnet_optimizer.step()
    new_weight_dict = {}
    for agent_id in curr_weights.keys():
        new_weights = hnet(int(agent_id))
        new_weight_dict[agent_id] = new_weights
    # Set weights of each worker, including remote replicas
    for w in agent.workers.remote_workers():
        w.set_weights.remote(new_weight_dict)
    #agent.set_weights(detach_weights(new_weight_dict))
    return result, new_weight_dict

def detach_weights(weights):
    new_weights = {}
    for agent_id, agent_params in weights.items():
        agent_param_dict = {}
        for k, v in agent_params.items():
            agent_param_dict[k] = v.detach()
        new_weights[agent_id] = agent_param_dict
    return new_weights

def train(agent, args):
    """
    Purpose: Train agent in environment. 
    """
    library = args.library
    algo = args.algo
    env = args.gym_env
    num_steps = args.num_steps

    if library == "rllib":
        print("Initializing Ray")
        # ray.init()
        print("Ray Initialized")

    ## Beginning Training ##
    print("Beginning training.")
    to_log = []
    training_steps = 0
    if args.gym_env in ["socialgame_multi", "microgrid_multi"] and args.use_hnet:
        weights = {}
        for agent_id in range(hnet.n_nodes):
            weights[str(agent_id)] = hnet(agent_id)
        # Set weights of each worker, including remote replicas
        #agent.workers.foreach_worker(lambda ev: ev.get_policy().set_weights(detach_weights(weights)))
        agent.set_weights(detach_weights(weights))
    #breakpoint()
    num_train_steps = 0
    while training_steps < num_steps:
        result = agent.train()
        training_steps = result["timesteps_total"]
        num_train_steps += 1
        if args.gym_env in ["socialgame_multi", "microgrid_multi"] and args.use_hnet and num_train_steps % args.hnet_num_local_steps == 0:
            print("****\nHNET UPDATE AT TIMESTEP {}\n****".format(training_steps))
            result, weights = pfl_hnet_update(agent, result, args, old_weights=weights)
        
        
        log = {name: result[name] for name in to_log}
        wandb.log(result['custom_metrics'])
        

    #list of different local agents
    #for each local agent, we call .train()
    #at the end of .train, that local agent should
    #have second order gradient updates ready for the hypernetwork
    #We should access those from out here and use them to update the hypernetwork
    #And then set the new local agent model weights

    #TODO: create a custom RLLib Policy and Trainer class that will do the optimization
    # on the local agent and compute second order gradient updates at the end of .train

    # callbacks.save() TODO: Implement callbacks
######################################
#### Arguments and Configurations ####
######################################

# Add environments here to be included when configuring an agent
environments = {
    "socialgame": SocialGameEnvRLLib,
    "microgrid": MicrogridEnvRLLib,
    "microgrid_multi": MultiAgentMicrogridEnv,
    "socialgame_multi": MultiAgentSocialGameEnv
}

parser = argparse.ArgumentParser()
# Custom Agents Arguments
parser.add_argument(
    "--custom_config",
    # type=argparse.FileType("r"),
    type=str,
    default = None,
    help="Path to custom config file"
)
# Library Argument
parser.add_argument(
    "--library",
    help = "What RL Library backend is in use",
    type = str,
    default = "rllib",
    choices = ["rllib", "tune"]
)
# Algorithm Arguments
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    type=str,
    default="ppo",
    choices=["sac", "ppo", "maml", "uc_bandit", "ccppo", "single_ppo"]
)
parser.add_argument(
    "--num_workers",
    help = "Number of cores you want to \
            sacrifice to RLLib to run envs in parallel",
    type=int,
    default=1
)
parser.add_argument(
    "--num_mg_workers", 
    help = "Number of cores each environment should dedicate to optimizing \
            building responses in microgrid",
    type=int,
    default=2
)
parser.add_argument(
    "--action_space_string",
    help="action space for algorithm",
    default="continuous",
)
parser.add_argument(
    "--policy_type",
    help="Type of Policy (e.g. MLP, LSTM) for algo",
    default="mlp",
    choices=["mlp", "lstm"],
)
parser.add_argument(
    "--reward_function",
    help="reward function to test",
    type=str,
    default="profit_maximizing",
    choices=["scaled_cost_distance", "log_cost_regularized", "log_cost", "scd", "lcr", "lc", "market_solving", "profit_maximizing"],
)

# Environment Arguments
parser.add_argument(
    "--gym_env", 
    help="Which Gym Environment you wish to use",
    type=str,
    choices=["socialgame", "microgrid", "microgrid_multi", "socialgame_multi"],
    default="microgrid_multi"
)
parser.add_argument(
    "--num_mg_optim_steps", 
    help = "Number of optim steps for mg to optimize building response",
    type=int,
    default=10
)
parser.add_argument(
    "--max_ep_steps", 
    help = "Number of steps in each episode",
    type=int,
    default=7 # week-long trajectory
)
parser.add_argument("--starting_day",
    help = "Which day for each environment to start on on resets",
    type=int,
    default=15)
parser.add_argument(
    "--env_id",
    help="Environment ID for Gym Environment",
    type=str,
    choices=["v0", "monthly"],
    default="v0",
)
parser.add_argument(
    "--response_type_string",
    help="Player response function (l = linear, t = threshold_exponential, s = sinusoidal",
    type=str,
    default="l",
    choices=["l", "t", "s"],
)
# Experiment Arguments
parser.add_argument(
    "--exp_name",
    help="experiment_name",
    type=str,
    default="experiment"
)
parser.add_argument(
    "--num_steps",
    help="Number of timesteps to train algo",
    type=int,
    default=50000,
)
parser.add_argument(
    "--energy_in_state",
    help="Whether to include energy in state (default = F)",
    action="store_true"
)
parser.add_argument(
    "--price_in_state",
    help="Whether to include price in state (default = F)",
    action="store_false"
)
parser.add_argument(
    "--points_multiplier",
    help="points multiplier for SocialGame simulated agents (default = 10)",
    default=10,
    type=float
)
parser.add_argument(
    "--batch_size",
    help="Batch Size for sampling from replay buffer",
    type=int,
    default=5,
    choices=[i for i in range(1, 30)],
)
parser.add_argument(
    "--one_day",
    help="Specific Day of the year to Train on (default = 15, train on day 15)",
    type=int,
    default=15,
    choices=[i for i in range(365)],
)
parser.add_argument(
    "--manual_tou_magnitude",
    help="Magnitude of the TOU during hours 5,6,7. Sets price in normal hours to 0.103.",
    type=float,
    default=.4
)
parser.add_argument(
    "--pricing_type",
    help="time of use or real time pricing",
    type=str,
    choices=["TOU", "RTP"],
    default="TOU",
)
parser.add_argument(
    "--number_of_participants",
    help="Number of players ([1, 20]) in social game",
    type=int,
    default=10,
    choices=[i for i in range(1, 21)],
)
parser.add_argument(
    "--learning_rate",
    help="learning rate of the the agent",
    type=float,
    default=3e-4,
)
# Logging Arguments
parser.add_argument(
    "-w",
    "--wandb",
    help="Whether to upload results to wandb. must have wandb key.",
    action="store_true"
)
parser.add_argument(
    # "--base_log_dir",
    "--log_path",
    help="Base directory for tensorboard logs",
    type=str,
    default="./logs/"
)
parser.add_argument(
    "--bulk_log_interval",
    help="Interval at which to save bulk log information",
    type=int,
    default=50000
)
parser.add_argument(
    "--bin_observation_space",
    help= "Whether to bin the observations.",
    action="store_true"
)
parser.add_argument(
    "--smirl_weight",
    help="Whether to run with SMiRL. When using SMiRL you must specify a weight.",
    type = float,
    default=None,
)
parser.add_argument(
    "--circ_buffer_size",
    help="Size of circular smirl buffer to use. Will use an unlimited size buffer in None",
    type = float,
    default=None,
)
# Machine Arguments
parser.add_argument(
    "-l",
    "--local_mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true"
)
parser.add_argument(
    "--num_inner_steps",
    help="Number of local model training steps",
    type= int,
    default=1
)
parser.add_argument(
    "--num_gpus",
    help="Number of gpus",
    type= int,
    default=0
)
parser.add_argument(
    "--scenarios",
    nargs='+',
    help="List of complex_batt_pv_scenarios to have separate agents for",
    default=[]
)
# Hypernetwork parameters
parser.add_argument(
    "--use_hnet",
    help="Whether or not to use the hypernetwork for MARL optimization",
    action="store_true",
    )
parser.add_argument(
    "--hnet_embedding_dim",
    help="Size of the embedding",
    type= int,
    default=1
)
parser.add_argument(
    "--hnet_num_layers",
    help="Number of linear layers",
    type= int,
    default=1
)
parser.add_argument(
    "--hnet_num_hidden",
    help="Size of the hidden layers",
    type= int,
    default=1
)
parser.add_argument(
    "--hnet_out_params_path",
    help="Path to a file containing a dict specifying the size of the output weights",
    type= str,
    default="./socialgame_param_dict.txt"
)
parser.add_argument(
    "--hnet_lr",
    help="Hnet learning rate",
    type= float,
    default=3e-4
)
parser.add_argument(
    "--hnet_num_local_steps",
    help = "Number of local steps between hnet updates",
    type=int,
    default=1
)
#
# Call get_agent and train to recieve the agent and then train it. 
#
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following options: {args}")

    # Uploading logs to wandb
    if args.wandb:
        wandb.init(project="energy-demand-response-game", entity="social-game-rl")
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)

    # Get Agent
    agent = get_agent(args)
    print("Agent initialized.")

    # Training
    print(f'Beginning Testing! Logs are being saved somewhere')
    train(agent, args)



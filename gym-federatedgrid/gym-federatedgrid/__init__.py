from gym.envs.registration import register

register(
    id="federated-v0",
    entry_point="gym_federatedgrid:FederatedEnv",
)

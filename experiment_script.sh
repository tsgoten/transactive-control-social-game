#!/bin/bash

python feudal_trainer.py -w --num_steps=50000 --gym_env=socialgame_env --lower_level_reward_type=directional --wandb_group=social_game_rewards &
s


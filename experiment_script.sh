#!/bin/bash

python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=directional -wandb_group=social_game_rewards &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=l1 -wandb_group=social_game_rewards &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=l2 -wandb_group=social_game_rewards &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=l1 -wandb_group=social_game_rewards

bash singularity_preamble_new.sh && python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=directional -wandb_group=social_game_rewards &
bash singularity_preamble_new.sh && python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=l1 -wandb_group=social_game_rewards &
bash singularity_preamble_new.sh && python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=l2 -wandb_group=social_game_rewards &
bash singularity_preamble_new.sh && python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_level_reward_type=l1 -wandb_group=social_game_rewards


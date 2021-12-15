bash singularity_preamble_new.sh
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_spatial --lower_level_reward_type=directional -wandb_group=microgrid_test &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_spatial --lower_level_reward_type=l1 -wandb_group=microgrid_test &

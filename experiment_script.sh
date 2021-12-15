bash singularity_preamble_new.sh
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_reward_function=directional &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_reward_function=l1 &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_reward_function=l2 &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_timewise --lower_reward_function=l1



#!/bin/bash
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_spatial --wandb_group=final_microgrid_test &
python feudal_trainer.py -w --num_steps=50000 --gym_env=feudal_spatial_lower_baseline --wandb_group=final_microgrid_test

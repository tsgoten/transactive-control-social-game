import os
for i in range(3):
    #os.system("python ExperimentRunner.py -w --gym_env=socialgame_multi --hnet_embedding_dim=338 --hnet_lr=0.0005578432454133512 --hnet_num_hidden=488 --hnet_num_layers=4 --custom_config=configs/even_spread.json --use_hnet --num_workers=8 --exp_name=hnet")
    os.system("python ExperimentRunner.py -w --gym_env=socialgame_multi --bulk_log_interval=500000 --custom_config=configs/even_spread.json --num_workers=8 --exp_name=hnet_vanilla --num_steps=500000")
    #os.system("python ExperimentRunner.py -w --gym_env=socialgame_multi --custom_config=configs/even_spread.json --num_workers=8 --algo=single_ppo --exp_name=hnet_singleppo")

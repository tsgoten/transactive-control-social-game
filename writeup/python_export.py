import wandb
import pandas as pd 
import numpy as np
import sys
import run_queries
X_KEY = "ray/tune/info/num_steps_sampled"
Y_KEY = "ray/tune/custom_metrics/agg/reward_mean_mean"
api = wandb.Api(timeout=20)
graph_name = sys.argv[1]
# Project is specified by <entity/project-name>
runs = api.runs("social-game-rl/energy-demand-response-game",
    run_queries.QUERIES[graph_name]
    )
api.runs("social-game-rl/energy-demand-response-game")
vals = {}
for run in runs:
    data_df = run.history(keys=[Y_KEY], samples=10000, x_axis=X_KEY)
    tag = run.config['tag']
    if tag not in vals:
        vals[tag] = data_df
    else:
        vals[tag] = pd.merge(vals[tag], data_df, on=X_KEY, how='outer', suffixes= [None, "_{}".format(len(vals[tag]))])
all_df = None
for tag, val in vals.items():
    curr_df = {X_KEY: val[X_KEY], 
                tag + ' - ' + Y_KEY : val.iloc[:, 1:].mean(axis=1), 
                tag + ' - ' + Y_KEY + '__std': val.iloc[:, 1:].std(axis=1), 
                tag + ' - ' + Y_KEY + '__ste': val.iloc[:, 1:].std(axis=1) / np.sqrt(len(val.keys())-1)}
    curr_df = pd.DataFrame.from_dict(curr_df)
    if all_df is None:
        all_df = curr_df
    else:
        all_df = pd.merge(all_df, curr_df, on=X_KEY, how='outer')
all_df = all_df.sort_values(by=X_KEY)


# all_df = pd.concat([name_df, config_df,summary_df], axis=1)
all_df.to_csv("data/{}.csv".format(graph_name), index=False)
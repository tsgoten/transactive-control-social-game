import matplotlib.pyplot as plt
import matplotlib
from graph_logs import *
import os
xticks = [2500 * i for i in range(5)]
def graph(file, window_width=5, smooth_baseline=True):
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "hnet" in name:
            return "PFL"
        elif "afl" in name:
            return "FedAvg"
        elif "baseline" in name:
            return "Baseline"
        else:
            return "ERROR"
    
    def color_func(name):
        if "PFL" in name:
            return "tab:blue"
        elif "FedAvg" in name:
            return "tab:red"
        elif "Baseline" in name:
            return "tab:green"
        else:
            return "black"

    data_func = lambda x: x
    
    yticks = [200 * i for i in range(4)]
    draw(file + ".csv", 
        TAG, 
        "Time (days)", 
        "Mean Profit per Microgrid", 
        fig_name=file,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        xticks=xticks,
        yticks=yticks,
        window_width=window_width, 
        data_func=data_func,
        color_func=color_func,
        smooth_baseline=smooth_baseline, 
        err_scale=2)
    
def graph_zeroshot(file, window_width=5, smooth_baseline=True):
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "zeroshot_hnet" in name:
            return "Few-Shot PFL"
        elif "hnet" in name:
            return "Original PFL"
        elif "baseline" in name:
            return "Baseline"
        else:
            return "ERROR"
    
    def color_func(name):
        if "Few-Shot" in name:
            return "tab:purple"
        elif "Original" in name:
            return "tab:blue"
        elif "Baseline" in name:
            return "tab:green"
        else:
            return "black"

    data_func = lambda x: x
    
    yticks = [200 * i for i in range(4)]
    draw(file + ".csv", 
        TAG, 
        "Time (days)", 
        "Mean Profit per Microgrid", 
        fig_name=file,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        xticks=xticks,
        yticks=yticks,
        window_width=window_width, 
        data_func=data_func,
        color_func=color_func,
        smooth_baseline=smooth_baseline)

def graph_zeroshot_all(file, window_width=5, smooth_baseline=True):
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "zeroshot_hnet" in name:
            _, b = name.split("custom")
            b, _ = b.split("_")
            print(b)
            return "Few-Shot PFL ({})".format(b)
        elif "hnet" in name:
            return "Original PFL"
        elif "baseline" in name:
            return "Baseline"
        else:
            return "ERROR"
    
    def color_func(name):
        if "5" in name:
            return "tab:red"
        if "10" in name:
            return "tab:orange"
        if "Few-Shot" in name and "20" in name:
            return "tab:purple"
        elif "Original" in name:
            return "tab:blue"
        elif "Baseline" in name:
            return "tab:green"
        else:
            return "black"

    data_func = lambda x: x
    
    yticks = [200 * i for i in range(4)]
    draw(file + ".csv", 
        TAG, 
        "Time (days)", 
        "Mean Profit per Microgrid", 
        fig_name=file,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        xticks=xticks,
        yticks=yticks,
        window_width=window_width, 
        data_func=data_func,
        color_func=color_func,
        smooth_baseline=smooth_baseline)

graph_zeroshot("zeroshot_20",  window_width=10, smooth_baseline=True)
graph_zeroshot_all("zeroshot_all",  window_width=10, smooth_baseline=True)

# graph("medium_10", smooth_baseline=False)
graph("simple_10", window_width=20, smooth_baseline=True)
graph("complex_10",  window_width=10, smooth_baseline=True)
graph("medium_20")
graph("simple_20", window_width=20, smooth_baseline=True)
graph("complex_20", window_width=10, smooth_baseline=True)
graph("simple_05", window_width=20, smooth_baseline=True)
graph("medium_05", window_width=10)
graph("complex_05", window_width=20, smooth_baseline=True)
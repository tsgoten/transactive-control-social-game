import matplotlib.pyplot as plt
import matplotlib
from graph_logs import *
import os

def graph(file, window_width=5, smooth_baseline=True):
    xticks = [2500 * i for i in range(5)]
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "hnet" in name:
            return "PFH"
        elif "afl" in name:
            return "FedAvg"
        elif "baseline" in name:
            return "Baseline"
        else:
            return "ERROR"
    
    def color_func(name):
        if "PFH" in name:
            return "tab:blue"
        elif "FedAvg" in name:
            return "tab:red"
        elif "Baseline" in name:
            return "tab:green"
        else:
            return "black"

    data_func = lambda x: x
    
    yticks = [200 * i for i in range(1, 4)]
    exp_type = None
    if "simple" in file:
        exp_type = "simple"
    elif "medium" in file:
        exp_type = "medium"
    elif "complex" in file:
        exp_type = "complex"
    draw("norl_{}.csv".format(exp_type), 
        TAG, 
        "Time (days)", 
        "Mean Microgrid Profit ($)", 
        fig_name=file,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "No RL", 
        xticks=xticks,
        yticks=yticks,
        window_width=window_width, 
        data_func=data_func,
        color_func=color_func,
        smooth_baseline=smooth_baseline, 
        plot_legend=False,
        err_scale=1,
        reset=False)
    draw(file + ".csv", 
        TAG, 
        "Time (days)", 
        "Mean Microgrid Profit ($)", 
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
        plot_legend=False,
        err_scale=2)
    
def graph_zeroshot(file, window_width=5, smooth_baseline=True):
    xticks = [1000 * i for i in range(5)]
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "zeroshot_hnet" in name:
            return "Few-Shot PFH"
        elif "hnet" in name:
            return "Original PFH"
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
        "Mean Microgrid Profit ($)", 
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
        cutoff=4000,
        legend_font=12)

def graph_zeroshot_all(file, window_width=5, smooth_baseline=True):
    xticks = [2500 * i for i in range(5)]
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "zeroshot_hnet" in name:
            _, b = name.split("custom")
            b, _ = b.split("_")
            return "Few-Shot PFH ({})".format(b)
        elif "hnet" in name:
            return "Original PFH"
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
    
    yticks = [200 * i for i in range(1, 4)]
    draw(file + ".csv", 
        TAG, 
        "Time (days)", 
        "Mean Microgrid Profit ($)", 
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
        legend_font=12)

def graph_zeroshot_comparison(file, window_width=10, smooth_baseline=True):
    xticks = [1000 * i for i in range(5)]
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=3
    axis_width=3
    def name_func(name):
        if "noembed" in name:
            return "Contextual PFH"
        elif "zeroshot_hnet" in name:
            return "Categorical PFH"
        elif "hnet" in name:
            return "Original PFH"
        else:
            return "ERROR"
    
    def color_func(name):
        if "Contextual" in name:
            return "tab:purple"
        elif "Original" in name:
            return "tab:blue"
        elif "Categorical" in name:
            return "tab:cyan"
        else:
            return "black"

    data_func = lambda x: x
    
    yticks = [200 * i for i in range(4)]
    draw(file + ".csv", 
        TAG, 
        "Time (days)", 
        "Mean Microgrid Profit ($)", 
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
        cutoff=4000,
        legend_font=12,
        include_errors=False)

# graph_zeroshot("zeroshot_20",  window_width=10, smooth_baseline=True)
# graph_zeroshot_all("zeroshot_all",  window_width=10, smooth_baseline=True)

# graph("medium_10", window_width=10, smooth_baseline=True)
# graph("simple_10", window_width=10, smooth_baseline=True)
# graph("complex_10",  window_width=10, smooth_baseline=True)
# graph("medium_20",  window_width=10, smooth_baseline=True)
# graph("simple_20", window_width=10, smooth_baseline=True)
# graph("complex_20", window_width=10, smooth_baseline=True)
# graph("simple_05", window_width=10, smooth_baseline=True)
# graph("medium_05", window_width=10, smooth_baseline=True)
# graph("complex_05", window_width=10, smooth_baseline=True)

graph_zeroshot_comparison("zeroshot_comparison", window_width=20)
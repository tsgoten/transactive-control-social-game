from turtle import color
import matplotlib.pyplot as plt
import matplotlib
from graph_logs import *
import os
import matplotlib
import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
default_colors = list(prop_cycle.by_key()['color']) + ['r', 'g', 'b']

def graph_fig1(path, name, TAG="Eval/AverageReturn"):
    
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=2
    axis_width=3
    data_func = lambda x: x
    colors = {}
    color_idx = 0
    def name_func(name):
        sections = name.split('_')
        for section in sections:
            if "opb" in section:
                opb = section[len("opb"):]
            if "adveps" in section:
                adveps = section[len("adveps"):]
                if adveps.strip() =="-1":
                    adveps="0.005"
                adveps=float(adveps) * 100
        return "ρ={}, ε={}%".format(opb, adveps)

    def color_func(name):
        nonlocal color_idx
        sections = name.split('_')
        for section in sections:
            if "opb" in section:
                opb = float(section[len("opb"):])
            if "adveps" in section:
                adveps = float(section[len("adveps"):])
        color = colors.get(opb, default_colors[color_idx])
        color_idx+=1
        colors[opb] = color
        return color

    def style_func(name):
        sections = name.split('_')
        for section in sections:
            if "opb" in section:
                opb = section[len("opb"):]
            if "adveps" in section:
                adveps = section[len("adveps"):]
        if str(adveps) == '-1':
            return '--'
        if str(adveps) == '0.1':
            return ':'
        else:
            return '-'

    draw(path, 
        TAG, 
        "Number of Steps", 
        "Average Return", 
        fig_name=name,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        window_width=5, 
        reset=True,
        data_func=data_func, 
        name_func=name_func,
        color_func=color_func,
        style_func=style_func)
    return colors

def graph_fig2(path, name, colors, TAG="Eval/AverageReturn"):
    
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=2
    axis_width=3
    data_func = lambda x: x
    color_idx = 0
    def name_func(name):
        adv_type = name[len("Adv_type: "):].capitalize()
        if adv_type.strip() == "Descent":
            adv_type = "FGSM"
        elif adv_type.strip() == "None":
            adv_type = "Vanilla"
        return adv_type

    def color_func(name):
        print(colors)
        nonlocal color_idx
        sections = path[:-4].split('_')
        for section in sections:
            if "opb" in section:
                print(section[len("opb")])
                opb = float(section[len("opb"):])
            if "eps" in section:
                adveps = float(section[len("eps"):])
        color = colors.get(opb, default_colors[color_idx])
        color_idx+=1
        colors[opb] = color
        adv_type = name[len("Adv_type: "):].capitalize()
        if adv_type == "Descent":
            return color
        elif adv_type == "None":
            return colors[0.0]
        else:
            return default_colors[-6]

    def style_func(name):
        sections = path[:-4].split('_')
        for section in sections:
            if "opb" in section:
                print(section[len("opb")])
                opb = float(section[len("opb"):])
            if "eps" in section:
                adveps = float(section[len("eps"):])
        if 'descent' not in name:
            return '-'
        if str(adveps) == '-1' or str(adveps=='-1.0'):
            return '--'
        if str(adveps) == '0.1':
            return ':'
        else:
            return '-'

    draw(path, 
        TAG, 
        "Number of Steps", 
        "Average Return", 
        fig_name=name,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        window_width=5, 
        reset=True,
        data_func=data_func, 
        name_func=name_func,
        color_func=color_func,
        style_func=style_func)

def graph_socialgame(path, name, TAG="Eval/AverageReturn"):
    
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=2
    axis_width=3
    data_func = lambda x: x
    colors = {}
    color_idx = 0
    def name_func(name):
        sections = name.split('_')
        for section in sections:
            if "vanilla" in section:
                apb=0
                adveps=0
                break
            if "apb" in section:
                apb = section[len("apb"):]
            if "eps" in section and "steps" not in section:
                adveps = section[len("eps"):]
                if adveps.strip() =="-1":
                    adveps="0.005"
                adveps=float(adveps) * 100
        return "ρ={}, ε={}%".format(apb, adveps)

    def color_func(name):
        nonlocal color_idx
        sections = name.split('_')
        for section in sections:
            if "vanilla" in section:
                opb=0
                adveps=0
                break
            if "apb" in section:
                opb = float(section[len("apb"):])
            if "eps" in section and "steps" not in section:
                adveps = float(section[len("eps"):])
        color = colors.get(opb, default_colors[color_idx])
        color_idx+=1
        colors[opb] = color
        return color

    def style_func(name):
        sections = name.split('_')
        for section in sections:
            if "vanilla" in section:
                opb=0
                adveps=0
                break
            if "apb" in section:
                opb = section[len("apb"):]
            if "eps" in section and "steps" not in section:
                adveps = section[len("eps"):]
        if str(adveps) == '-1':
            return '--'
        if str(adveps) == '0.1':
            return ':'
        else:
            return '-'
        
    def data_func(x):
        x = np.exp(-x)
        x /= 75
        x *= -500 * 0.001
        return x

    draw(path, 
        TAG, 
        "Number of Steps", 
        "Average Return = Negative of Daily Energy Cost", 
        fig_name=name,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        window_width=5, 
        reset=True,
        data_func=data_func, 
        name_func=name_func,
        color_func=color_func,
        style_func=style_func,
        legend_loc="upper left")
    return colors

def graph_defense(path, name, TAG="Eval/AverageReturn"):
    
    FIGS_DIR = "figs/"
    FORMAT='png'
    linewidth=2
    axis_width=3
    data_func = lambda x: x
    colors = {}
    color_idx = 0
    def name_func(name):
        name = name.split("-")[0].split(":")[1].strip()
        if name == "filter0.1":
            opb = 0.5
            adveps = 10
            filt="With Filter"
        elif name == "nofilter":
            opb = 0.5
            adveps = 10
            filt="Without Filter"
        else:
            opb = 0.0
            adveps = 0.0
            filt="Baseline"
        return "ρ={}, ε={}%, {}".format(opb, adveps, filt)

    def color_func(name):
        nonlocal color_idx
        name = name.split("-")[0].split(":")[1].strip()
        if name == "filter0.1":
            opb = 0.5
            adveps = 10
            filt="With Filter"
            color="green"
        elif name == "nofilter":
            opb = 0.5
            adveps = 10
            filt="Without Filter"
            color="orange"
        else:
            opb = 0.0
            adveps = 0.0
            filt="Baseline"
        #color = colors.get(opb, default_colors[color_idx])
        color_idx+=1
        if filt == "Baseline":
            color="red"
        #colors[opb] = color
        
        return color

    def style_func(name):
        sections = name.split('_')
        for section in sections:
            if "opb" in section:
                opb = section[len("opb"):]
            if "adveps" in section:
                adveps = section[len("adveps"):]
        if str(adveps) == '-1':
            return '--'
        if str(adveps) == '0.1':
            return ':'
        else:
            return '-'

    draw(path, 
        TAG, 
        "Number of Steps", 
        "Average Return", 
        fig_name=name,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        window_width=5, 
        reset=True,
        data_func=data_func, 
        name_func=name_func,
        color_func=color_func,)
        #style_func=style_func)
    return colors
colors = graph_fig1("fig1.csv", "fig1")
graph_fig2("fig2_eps-1_opb0.5.csv", "fig2_eps-1_opb0.5", colors)
graph_fig2("fig2_eps0.1_opb0.5.csv", "fig2_eps0.1_opb0.5", colors)
graph_fig2("fig2_eps0.1_opb1.csv", "fig2_eps0.1_opb1", colors)


TAG = "Train/AverageReturn"
colors = graph_fig1("fig8.csv", "fig8", TAG)
graph_fig2("fig9_eps1.0_opb5.csv", "fig9_eps1.0_opb5", colors, TAG=TAG)
graph_fig2("fig9_eps1.0_opb10.csv", "fig9_eps1.0_opb10", colors, TAG=TAG)
graph_fig2("fig9_eps1.0_opb20.csv", "fig9_eps1.0_opb20", colors, TAG=TAG)
graph_fig2("fig11_eps1.0_opb5.csv", "fig11_eps1.0_opb5", colors, TAG=TAG)


colors = graph_socialgame("fig12.csv", "fig12", )

graph_defense("fig13.csv", "fig13")
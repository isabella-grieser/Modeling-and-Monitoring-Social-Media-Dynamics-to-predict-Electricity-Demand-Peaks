from data.social_media.process_sqlite import *
from framework import EstimationFramework
import matplotlib.pyplot as plt
import matplotlib.dates as md
from statistics import mean
import numpy as np
from utils.utils import running_mean
from frameworkrun import create_plot

def plot_basic_timeline(ax1, ax2, x_all, y_vals, y_ref, s_true, i_true, r_true, ls, color,
                        label, start_index=0, end_index=-1):
    y_max, y_min, y_average = [], [], []
    for i in range(len(y_vals[0])):
        y_max.append(max(y_vals[j][i] for j in range(len(y_vals))))
        y_min.append(min(y_vals[j][i] for j in range(len(y_vals))))
        y_average.append(mean(y_vals[j][i] for j in range(len(y_vals))))

    if end_index < 0:
        x_all, y_ref = x_all[start_index:end_index], y_ref[start_index:end_index]
        y_max, y_min, y_average = y_max[start_index:end_index], y_min[start_index:end_index], y_average[
                                                                                              start_index:end_index]
    else:
        x_all, y_ref = x_all[start_index:], y_ref[start_index:]
        y_max, y_min, y_average = y_max[start_index:], y_min[start_index:], y_average[start_index:]

    # ax1.fill_between(x_all, y_min, y_max, color='blue', alpha=.5, linewidth=0)
    ax1.plot(x_all, y_average, linewidth=2, color=color, ls=ls, label=label)

    s_max, s_min, s_average = [], [], []
    i_max, i_min, i_average = [], [], []
    r_max, r_min, r_average = [], [], []
    for i in range(len(s_true[0])):
        s_max.append(max(s_true[j][i] for j in range(len(s_true))))
        s_min.append(min(s_true[j][i] for j in range(len(s_true))))
        s_average.append(mean(s_true[j][i] for j in range(len(s_true))))

        i_max.append(max(i_true[j][i] for j in range(len(i_true))))
        i_min.append(min(i_true[j][i] for j in range(len(i_true))))
        i_average.append(mean(i_true[j][i] for j in range(len(i_true))))

        r_max.append(max(r_true[j][i] for j in range(len(r_true))))
        r_min.append(min(r_true[j][i] for j in range(len(r_true))))
        r_average.append(mean(r_true[j][i] for j in range(len(r_true))))
    if end_index < 0:
        s_max, s_min, s_average = s_max[start_index:end_index], s_min[start_index:end_index], s_average[
                                                                                              start_index:end_index]
        i_max, i_min, i_average = i_max[start_index:end_index], i_min[start_index:end_index], i_average[
                                                                                              start_index:end_index]
        r_max, r_min, r_average = r_max[start_index:end_index], r_min[start_index:end_index], r_average[
                                                                                              start_index:end_index]
        start_index = next(ind for ind, ia in enumerate(i_average) if ia > 1.5)
        i_start_proc = i_average[start_index-1:]
        print(
            f"start slope for {label}: {(i_start_proc[2] - i_start_proc[0]) / 0.5} "
            f"between {x_all[start_index-1]} and {x_all[start_index+1]}")
    else:
        s_max, s_min, s_average = s_max[start_index:], s_min[start_index:], s_average[start_index:]
        i_max, i_min, i_average = i_max[start_index:], i_min[start_index:], i_average[start_index:]
        r_max, r_min, r_average = r_max[start_index:], r_min[start_index:], r_average[start_index:]

    # ax2.fill_between(x_all, s_min, s_max, color='green', alpha=.5, linewidth=0)
    # ax2.plot(x_all, s_average, linewidth=2, color='green', ls=ls, label="susceptible")

    # ax2.fill_between(x_all, i_min, i_max, color='red', alpha=.5, linewidth=0)
    ax2.plot(x_all, i_average, linewidth=2, color=color, ls=ls, label="infection process")
    # ax2.plot(x_all, i_average, linewidth=2, color='red', ls=ls, label="infected")
    # ax2.fill_between(x_all, r_min, r_max, color='blue', alpha=.5, linewidth=0)
    #ax2.plot(x_all, r_average, linewidth=2, color='blue', ls=ls, label="removed")

    ax1.plot(x_all, y_ref, linewidth=2, color='black', label="reference")

if __name__ == "__main__":
    data = get_groenwald()

    start = datetime(2022, 11, 20, 16, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)
    action_start = datetime(2022, 11, 20, 19, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    st, e = 0, -1
    values = data.groupby(pd.Grouper(key="date", freq="30min"))["tweet"].count()[st:e]

    window = 3

    values_filtered = running_mean(values.values, window)
    index = values.index[int(window/2):-int(window/2)]

    with open("./config/demand-response.json", "r") as f:
        config = json.load(f)

    x_all, y_ref = None, None
    framework = None
    #p_vals = [4, 8, 12, 16, len(values_filtered)]
    #labels = ["after 2h", "after 4h", "after 6h", "after 8h", "full dataset"]
    #dots = ['--', '-.', ':', (0, (3, 5, 1, 5, 1, 5)), '-']
    #colors = ['gold', 'lawngreen', 'darkolivegreen', 'orange', 'blue']

    #p_vals = [4, 8, 12, 16, len(values_filtered)]
    #labels = ["after 2h", "after 4h", "after 6h", "after 8h", "full dataset"]
    #dots = ['--', '-.', ':', (0, (3, 5, 1, 5, 1, 5)), '-']
    #colors = ['gold', 'lawngreen', 'darkolivegreen', 'orange', 'blue']
    p_vals = [len(values_filtered)]
    labels = ["est. power consumption"]
    dots = ['-']
    colors = ['blue']
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    xfmt = md.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_ylabel(f"Power Demand (kW)", fontsize=12)
    ax1.set_xlabel(f"Time", fontsize=12)
    ax1.legend(loc="upper left")

    ax2.set_ylabel(f"Number of infected entities", fontsize=12)
    ax2.set_ylim([0, config["network"]["nodes"]])
    ax2.set_xlabel(f"Time", fontsize=12)
    xfmt2 = md.DateFormatter('%H:%M')
    ax2.xaxis.set_major_formatter(xfmt2)
    # ax2.set(xlim=(self.x[0], self.x[self.timespan + iterations]), ylim=(0, n_size))
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper right")

    for p, ls, c, lab in zip(p_vals, dots, colors, labels):
        y_vals, s_vals, i_vals, r_vals = [], [], [], []
        values_pred = values_filtered[:p]
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, year=2022, plot=False)

            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, iterations=100,
                                                y_max=1000, data=values_pred, minutes=30, estimation_end_time=len(index))
            y_vals.append(y_true)
            s_vals.append(s_true)
            i_vals.append(i_true)
            r_vals.append(r_true)
        plot_basic_timeline(ax1, ax2, x_all, y_vals=y_vals, y_ref=y_ref, label=lab,
                            s_true=s_vals, i_true=i_vals, r_true=r_vals, ls=ls, color=c,
                            start_index=85, end_index=-55)

    #ax1.axhline(framework.threshold, color='red', label="threshold")

    # ax1.axvline(x=start, color='green', ls=':', label='spread start')
    ax1.axvline(x=action_start, color='red', ls=':', label='action start')

    handles, labels = ax2.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax2.legend(handles, labels)

    ax2.legend(loc="upper right")
    ax1.legend(loc="upper right")

    handles, labels = ax1.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax1.legend(handles, labels)

    plt.show()

    fig1.savefig('images/power_demand.pdf',bbox_inches='tight', format="pdf")
    fig2.savefig('images/sir.pdf',bbox_inches='tight', format="pdf")
    # model prediction framework
    datapoints = range(2, int(len(values_filtered)), 4)
    y_max = []

    for p in datapoints:
        values_pred = values_filtered[:p]
        framework = EstimationFramework(config, year=2022, plot=False)

        x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
            framework.estimate_power_outage(start, action_start=action_start, iterations=100,
                                            y_max=1000, data=values_pred, minutes=30, estimation_end_time=len(index))
        diffs = [y_true[i] - y_ref[i] for i in range(len(y_true))]
        y_max.append(y_true[np.argmax(diffs)])

    plt.plot(datapoints, y_max)
    plt.xlabel("Number of datapoints", fontsize=12)
    plt.ylabel("Maximum estimated power consumption in kW", fontsize=12)
    plt.axhline(framework.threshold, color='red', label="threshold")
    plt.show()

    # plot different adoption rates for evs
    ev_adoption = [0, 0.25, 0.5, 0.75, 1]

    power_vals = []
    for p in ev_adoption:
        values_pred = values_filtered[:len(values_filtered)]
        config["model_args"]["electric_car"]["p"] = p
        vals = []
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, year=2022, plot=False)

            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, iterations=100,
                                                y_max=1000, data=values_pred, minutes=30, estimation_end_time=len(index))
            vals.append(y_true)

        average_val = []
        for i in range(len(vals[0])):
            average_val.append(mean(vals[j][i] for j in range(len(vals))))
        power_vals.append(average_val)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    start_i, end_i = 90, -55
    x = x_all[start_i:end_i]

    dots = ['--', '-.', 'dotted', '-', ":", "dashdot"]
    colors = ['gold', 'lawngreen', 'darkolivegreen', 'blue', "red", "gray"]

    for p, v, d in zip(ev_adoption, power_vals, dots):
        ax.plot(x, v[start_i:end_i], label=f"adoption rate={p}", ls=d)
    ax.plot(x, y_ref[start_i:end_i], color="black", label=f"reference")
    ax.set_ylabel("Power consumption in kW", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    # ax.axhline(framework.threshold, color='red', label="power threshold")
    ax.legend(loc="upper right")
    xfmt = md.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    fig2.savefig('images/variant_ev.pdf',bbox_inches='tight', format="pdf")
    plt.show()
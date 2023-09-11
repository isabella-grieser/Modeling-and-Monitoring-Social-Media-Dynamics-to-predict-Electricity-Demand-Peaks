import json

import matplotlib
from data.social_media.process_sqlite import *
from scipy import signal
from datetime import datetime
import datetime as dt
from framework import EstimationFramework
import matplotlib.dates as md
import matplotlib.pyplot as plt
from statistics import mean
import pytz

matplotlib.use("TkAgg")


def create_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    xfmt = md.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_ylabel(f"Power Consumption in kW")
    ax1.set_xlabel(f"Time")
    ax1.legend(loc="upper left")

    ax2.set_ylabel(f"Infection Process")
    ax2.set_xlabel(f"Time")
    xfmt2 = md.DateFormatter('%H:%M')
    ax2.xaxis.set_major_formatter(xfmt2)
    # ax2.set(xlim=(self.x[0], self.x[self.timespan + iterations]), ylim=(0, n_size))
    ax2.legend(loc="upper left")

    return fig, ax1, ax2


def plot_basic_timeline(x_all, y_vals, y_ref, s_true, i_true, r_true, power_tresh, spread=None, action=None):
    fig, ax1, ax2 = create_plot()
    ax1.plot(x_all, y_ref)
    y_max, y_min, y_average = [], [], []
    for i in range(len(y_vals[0])):
        y_max.append(max(y_vals[j][i] for j in range(len(y_vals))))
        y_min.append(min(y_vals[j][i] for j in range(len(y_vals))))
        y_average.append(mean(y_vals[j][i] for j in range(len(y_vals))))

    ax1.fill_between(x_all, y_min, y_max, color='blue', alpha=.5, linewidth=0)
    ax1.plot(x_all, y_average, linewidth=2, color='blue', label="true consumption")
    ax1.plot(x_all, y_ref, linewidth=2, color='black', label="ref consumption")
    ax1.axhline(power_tresh, color='red', label="power threshold")
    if spread is not None:
        ax1.axvline(x=spread, color='yellow', ls=':', label='spread start')
    if action is not None:
        ax1.axvline(x=action, color='red', ls=':', label='action start')

    ax1.legend(loc="upper right")

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

    ax2.fill_between(x_all, s_min, s_max, color='green', alpha=.5, linewidth=0)
    ax2.plot(x_all, s_average, linewidth=2, color='green', label="susceptible")
    ax2.fill_between(x_all, i_min, i_max, color='red', alpha=.5, linewidth=0)
    ax2.plot(x_all, i_average, linewidth=2, color='red', label="infected")
    ax2.fill_between(x_all, r_min, r_max, color='blue', alpha=.5, linewidth=0)
    ax2.plot(x_all, r_average, linewidth=2, color='blue', label="removed")
    ax2.legend(loc="upper left")
    plt.show()


def scenario1():
    data = get_typhoon_data()

    s_index, e_index = 0, -1
    values = data.groupby(pd.Grouper(key="date", freq="15min"))["tweet"].count()

    start = values.index[s_index]

    action_start = datetime(2013, 11, 7, 19, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)
    y = signal.savgol_filter(values.values, 53, 3)[s_index: e_index]

    with open("./config/demand-response.json", "r") as f:
        config = json.load(f)

    # plot basic timeline plots
    y_vals, s_vals, i_vals, r_vals = [], [], [], []
    x_all, y_ref = None, None

    framework = None
    for s in config["seeds"]:
        config["seed"] = s
        framework = EstimationFramework(config, plot=False)

        x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
            framework.estimate_power_outage(start, action_start=action_start, y_max=1000, data=y)
        y_vals.append(y_true)
        s_vals.append(s_true)
        i_vals.append(i_true)
        r_vals.append(r_true)

    # creation of the basic plot
    plot_basic_timeline(x_all, y_vals, y_ref, s_vals, i_vals, r_vals,
                        framework.threshold, spread=start, action=action_start)

    beta_vals = [0.1, 0.2, 0.3, 0.4]
    beta_results = []
    for j in beta_vals:
        vals = []
        for s in config["seeds"]:
            config["seed"] = s
            config["sim"]["beta"] = j

            framework = EstimationFramework(config, plot=False)

            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, y_max=1000, data=y)
            vals.append(max(y_true))
        beta_results.append(mean(vals))


def scenario2():
    start = None
    with open("./config/conspiracy.json", "r") as f:
        config = json.load(f)

    action_start = datetime(2013, 11, 7, 17, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    y_vals, s_vals, i_vals, r_vals = [], [], [], []
    x_all, y_ref = None, None
    for s in config["seeds"]:
        config["seed"] = s
        framework = EstimationFramework(config, year=2023)
        x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
            framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
        y_vals.append(y_true)
        s_vals.append(s_true)
        i_vals.append(i_true)
        r_vals.append(r_true)


def scenario3():
    start = None
    with open("./config/conspiracy.json", "r") as f:
        config = json.load(f)
    framework = EstimationFramework(config)
    framework.estimate_power_outage(start)


if __name__ == "__main__":
    scenario1()
    # scenario2()
    # scenario3()

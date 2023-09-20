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
import numpy as np

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


def plot_basic_timeline(x_all, y_vals, y_ref, s_true, i_true, r_true, power_tresh, spread=None, action=None,
                        start_index=0, end_index=-1):
    fig, ax1, ax2 = create_plot()
    y_max, y_min, y_average = [], [], []
    for i in range(len(y_vals[0])):
        y_max.append(max(y_vals[j][i] for j in range(len(y_vals))))
        y_min.append(min(y_vals[j][i] for j in range(len(y_vals))))
        y_average.append(mean(y_vals[j][i] for j in range(len(y_vals))))

    x_all, y_ref = x_all[start_index:end_index], y_ref[start_index:end_index]
    y_max, y_min, y_average = y_max[start_index:end_index], y_min[start_index:end_index], y_average[start_index:end_index]

    ax1.plot(x_all, y_ref)
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

    s_max, s_min, s_average = s_max[start_index:end_index], s_min[start_index:end_index], s_average[start_index:end_index]
    i_max, i_min, i_average = i_max[start_index:end_index], i_min[start_index:end_index], i_average[start_index:end_index]
    r_max, r_min, r_average = r_max[start_index:end_index], r_min[start_index:end_index], r_average[start_index:end_index]

    ax2.fill_between(x_all, s_min, s_max, color='green', alpha=.5, linewidth=0)
    ax2.plot(x_all, s_average, linewidth=2, color='green', label="susceptible")
    ax2.fill_between(x_all, i_min, i_max, color='red', alpha=.5, linewidth=0)
    ax2.plot(x_all, i_average, linewidth=2, color='red', label="infected")
    ax2.fill_between(x_all, r_min, r_max, color='blue', alpha=.5, linewidth=0)
    ax2.plot(x_all, r_average, linewidth=2, color='blue', label="removed")
    ax2.legend(loc="upper left")
    plt.show()


def basic_plot(config, start, action_start, iterations=200, y=None, start_index=0, end_index=-1):
    y_vals, s_vals, i_vals, r_vals, y_max = [], [], [], [], []
    x_all, y_ref = None, None
    framework = None
    for s in config["seeds"]:
        config["seed"] = s
        framework = EstimationFramework(config, plot=False)

        x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
            framework.estimate_power_outage(start, action_start=action_start, iterations=iterations, y_max=1000, data=y)
        y_vals.append(y_true)
        s_vals.append(s_true)
        i_vals.append(i_true)
        r_vals.append(r_true)
        diffs = [y_true[i] - y_ref[i] for i in range(len(y_true))]
        y_max.append(max(diffs))

    print(f"excess consumption max: {mean(y_max)}")

    plot_basic_timeline(x_all, y_vals, y_ref, s_vals, i_vals, r_vals,
                        framework.threshold, spread=start, action=action_start,
                        start_index=start_index, end_index=end_index)


def plot_vals(config, attr1, attr2, probs, start, action_start, iterations=200):
    final_vals = []
    i_average = []
    r_average = []
    print(f"change {attr2}")
    x = None
    for j in probs:
        vals = []
        all_i, all_r = [], []
        config[attr1][attr2] = j
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, plot=False)
            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, iterations=iterations, y_max=1000)
            max_val = max(y_true)
            x = x_all
            print(f"max y value: {max_val}")
            vals.append(max_val)
            all_i.append(i_true)
            all_r.append(r_true)

        average_i, average_r = [], []
        for i in range(len(all_i[0])):
            average_i.append(mean(all_i[j][i] for j in range(len(all_i))))
            average_r.append(mean(all_r[j][i] for j in range(len(all_r))))
        i_average.append(average_i)
        r_average.append(average_r)
        final_vals.append(mean(vals))
    return x, final_vals, i_average, r_average


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

    def plot_alpha(alpha_p):
        x, alpha_vals, i_average, r_average = plot_vals(config, "sim",
                                                        "alpha", alpha_p, start, action_start)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        start_i, end_i = 80, -150
        x = x[start_i: end_i]
        for r_val, i_val, b in zip(r_average, i_average, alpha_p):
            ax1.plot(x, i_val[start_i: end_i], label=f"alpha={b}")
            ax2.plot(x, r_val[start_i: end_i], label=f"alpha={b}")
        ax2.legend(loc="lower left")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of infected entities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of recovered entities")
        ax1.tick_params(labelrotation=45)
        ax2.tick_params(labelrotation=45)
        xfmt = md.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        plt.show()
        return alpha_p, alpha_vals

    def plot_beta(beta_p):
        x, beta_vals, i_average, r_average = plot_vals(config, "sim",
                                                       "beta", beta_p, start, action_start)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        start_i, end_i = 80, -150
        x = x[start_i: end_i]
        for r_val, i_val, b in zip(r_average, i_average, beta_p):
            ax1.plot(x, i_val[start_i: end_i], label=f"beta={b}")
            ax2.plot(x, r_val[start_i: end_i], label=f"beta={b}")
        ax2.legend(loc="lower left")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of infected entities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of recovered entities")
        ax1.tick_params(labelrotation=45)
        ax2.tick_params(labelrotation=45)
        xfmt = md.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        plt.show()
        return beta_p, beta_vals

    def plot_verify(verify_p):
        x, verify_vals, i_average, r_average = plot_vals(config, "sim",
                                                         "p_verify", verify_p, start, action_start)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        start_i, end_i = 80, -150
        x = x[start_i: end_i]
        for r_val, i_val, b in zip(r_average, i_average, verify_p):
            ax1.plot(x, i_val[start_i: end_i], label=f"p_verify={b}")
            ax2.plot(x, r_val[start_i: end_i], label=f"p_verify={b}")
        ax2.legend(loc="lower left")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of infected entities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of recovered entities")
        ax1.tick_params(labelrotation=45)
        ax2.tick_params(labelrotation=45)
        xfmt = md.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        plt.show()
        return verify_p, verify_vals

    def analyze_propagation(alpha_p, beta_p, verify_p):
        config["sim"]["beta"] = .2
        config["sim"]["alpha"] = .4
        config["sim"]["p_verify"] = .2
        alpha_vals, alpha_res = plot_alpha(alpha_p)
        config["sim"]["alpha"] = .4
        beta_vals, beta_res = plot_beta(beta_p)
        config["sim"]["beta"] = .2
        verify_vals, verify_res = plot_verify(verify_p)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
        ax1.plot(alpha_vals, alpha_res)
        ax2.plot(beta_vals, beta_res)
        ax3.plot(verify_vals, verify_res)
        ax1.set_ylabel("Maximum power consumption")
        ax1.set_xlabel("alpha")
        ax2.set_xlabel("beta")
        ax3.set_xlabel("p_verify")
        ax1.tick_params(labelrotation=45)
        ax2.tick_params(labelrotation=45)
        ax3.tick_params(labelrotation=45)
        xfmt = md.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        plt.show()

    def analyze_acting_params(acting_p, usage_p, available_p):
        acting_v, usage_v, available_v = [], [], []
        print("change will act")
        config["sim"]["beta"] = .5
        config["sim"]["alpha"] = .5
        config["sim"]["p_verify"] = .05

        config["network"]["available"] = 1
        config["sim"]["power_usage"] = 1
        x, y_reference = None, None
        for p in acting_p:
            y_trues = []
            config["sim"]["p_will_act"] = p
            for s in config["seeds"]:
                config["seed"] = s
                framework = EstimationFramework(config, plot=False)
                x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                    framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
                x = x_all
                y_reference = y_ref
                max_val = max(y_true)
                print(f"max y value: {max_val}")
                y_trues.append(y_true)

            y_average = []
            for i in range(len(y_trues[0])):
                y_average.append(mean(y_trues[j][i] for j in range(len(y_trues))))
            acting_v.append(y_average)

        config["sim"]["p_will_act"] = 1
        for p in usage_p:
            y_trues = []
            config["sim"]["power_usage"] = p
            for s in config["seeds"]:
                config["seed"] = s
                framework = EstimationFramework(config, plot=False)
                x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                    framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
                max_val = max(y_true)
                print(f"max y value: {max_val}")
                y_trues.append(y_true)

            y_average = []
            for i in range(len(y_trues[0])):
                y_average.append(mean(y_trues[j][i] for j in range(len(y_trues))))
            usage_v.append(y_average)

        config["sim"]["p_will_act"] = 1
        config["sim"]["power_usage"] = 1
        for p in available_p:
            y_trues = []
            config["network"]["available"] = p
            for s in config["seeds"]:
                config["seed"] = s
                framework = EstimationFramework(config, plot=False)
                x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                    framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
                max_val = max(y_true)
                print(f"max y value: {max_val}")
                y_trues.append(y_true)

            y_average = []
            for i in range(len(y_trues[0])):
                y_average.append(mean(y_trues[j][i] for j in range(len(y_trues))))
            available_v.append(y_average)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
        # cut parts of the plot
        start_i, end_i = 130, -130
        x = x[start_i: end_i]
        for usage_val, u_p in zip(usage_v, usage_p):
            # calc the difference
            y_plot = np.array(usage_val) - np.array(y_reference)
            ax1.plot(x, y_plot[start_i: end_i], label=f"power_usage={u_p}")

        for acting_val, a_p in zip(acting_v, acting_p):
            y_plot = np.array(acting_val) - np.array(y_reference)
            ax2.plot(x, y_plot[start_i: end_i], label=f"p_will_act={a_p}")

        for available_val, a_p in zip(available_v, available_p):
            y_plot = np.array(available_val) - np.array(y_reference)
            ax3.plot(x, y_plot[start_i: end_i], label=f"available={a_p}")

        # ax1.plot(x, y_reference, color='black', label="ref consumption")
        # ax2.plot(x, y_reference, color='black', label="ref consumption")
        xfmt = md.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_formatter(xfmt)
        ax1.set_ylabel("Power consumption in kW")
        ax1.set_xlabel("p_will_act")
        ax2.set_xlabel("power_usage")
        ax3.set_xlabel("available")
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper left")
        ax1.tick_params(labelrotation=45)
        ax2.tick_params(labelrotation=45)
        ax3.tick_params(labelrotation=45)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for usage_val, u_p in zip(usage_v, usage_p):
            ax.plot(x, usage_val[start_i: end_i], label=f"power_usage={u_p}")
        ax.plot(x, y_reference[start_i: end_i], color='black', label="ref consumption")
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_ylabel("Power consumption in kW")
        ax.set_xlabel("Time")
        ax.legend(loc="upper right")
        plt.xticks(rotation=45)
        plt.show()

    basic_plot(config, start, action_start, y)
    analyze_propagation([0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.99], [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.99],
                        [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5])
    analyze_acting_params(acting_p=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 1], usage_p=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
                          available_p=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])


def scenario2():
    with open("./config/conspiracy.json", "r") as f:
        config = json.load(f)

    data = get_typhoon_data()

    s_index, e_index = 0, -1
    values = data.groupby(pd.Grouper(key="date", freq="15min"))["tweet"].count()

    start = values.index[s_index]

    action_start = datetime(2013, 11, 7, 19, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)
    y = signal.savgol_filter(values.values, 53, 3)[s_index: e_index]

    basic_plot(config, start, action_start, iterations=200)

    config["fringe"] = False
    basic_plot(config, start, action_start, iterations=200)



def scenario3():
    with open("./config/wildfire.json", "r") as f:
        config = json.load(f)

    start = datetime(2013, 11, 6, 15, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    action_start = datetime(2013, 11, 6, 18, 15, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    basic_plot(config, start, action_start, iterations=200)


def scenario4():
    with open("./config/chemicalaccident.json", "r") as f:
        config = json.load(f)

    action_start = datetime(2013, 11, 6, 18, 30, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    basic_plot(config, action_start, action_start=None, iterations=100, start_index=90, end_index=-70)

    probs_p = [0, 0.03, 0.05, 0.1, 0.15, 0.2]
    probs_v = []
    for p in probs_p:
        y_trues = []
        config["model_args"]["heat_pump"]["p"] = p
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, plot=False)
            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(action_start, y_max=1000)
            diffs = [y_true[i] - y_ref[i] for i in range(len(y_true))]
            max_val = max(diffs)
            print(f"max y value: {max_val}")
            y_trues.append(max_val)
        probs_v.append(mean(y_trues))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    probs = [f"{int(p*100)}%"for p in probs_p]
    ax.bar(probs, probs_v, width=1, label=probs, edgecolor="white", linewidth=0.7)
    ax.set_ylabel("Additional power consumption in kW")
    ax.set_xlabel("Percentage of households with heat pumps")
    plt.xticks(rotation=45)
    plt.show()

    duration_p = [1, 2, 3, 4]
    duration_v = []
    config["model_args"]["heat_pump"]["p"] = 0.03
    for p in duration_p:
        y_trues = []
        config["model_args"]["heat_pump"]["duration"] = p
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, plot=False)
            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(action_start, y_max=1000)
            diffs = [y_true[i] - y_ref[i] for i in range(len(y_true))]
            max_val = max(diffs)
            print(f"max y value: {max_val}")
            y_trues.append(max_val)
        duration_v.append(mean(y_trues))

    print(f"durations: {duration_p}, results: {duration_v}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    duration = [str(d) for d in duration_p]
    ax.bar(duration, duration_v, width=1, label=duration, edgecolor="white", linewidth=0.7)
    ax.set_ylabel("Additional power consumption in kW")
    ax.set_xlabel("Duration of the usage of heat pumps")
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    # scenario1()
    # scenario2()
    scenario3()
    # scenario4()

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


def basic_plot(config, start, action_start, y=None):
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
    plot_basic_timeline(x_all, y_vals, y_ref, s_vals, i_vals, r_vals,
                        framework.threshold, spread=start, action=action_start)


def plot_vals(config, attr1, attr2, probs, start, action_start):
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
                framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
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
        config["sim"]["beta"] = .5
        config["sim"]["alpha"] = .5
        config["sim"]["p_verify"] = .01
        x, alpha_vals, i_average, r_average = plot_vals(config, "sim",
                                                        "alpha", alpha_p, start, action_start)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for r_val, i_val, b in zip(r_average, i_average, alpha_p):
            ax1.plot(x, i_val, label=f"alpha={b}")
            ax2.plot(x, r_val, label=f"alpha={b}")
        ax2.legend(loc="upper left")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("number of infected entities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("number of recovered entities")
        plt.show()
        return alpha_p, alpha_vals

    def plot_beta(beta_p):
        config["sim"]["beta"] = .5
        config["sim"]["alpha"] = .5
        config["sim"]["p_verify"] = .01
        x, beta_vals, i_average, r_average = plot_vals(config, "sim",
                                                       "beta", beta_p, start, action_start)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for r_val, i_val, b in zip(r_average, i_average, beta_p):
            ax1.plot(x, i_val, label=f"beta={b}")
            ax2.plot(x, r_val, label=f"beta={b}")
        ax2.legend(loc="upper left")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("number of infected entities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("number of recovered entities")
        plt.show()
        return beta_p, beta_vals

    def plot_verify(verify_p):
        config["sim"]["beta"] = .5
        config["sim"]["alpha"] = .5
        config["sim"]["p_verify"] = .5
        x, verify_vals, i_average, r_average = plot_vals(config, "sim",
                                                         "p_verify", verify_p, start, action_start)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # plt.plot(beta_vals, beta_results)
        for r_val, i_val, b in zip(r_average, i_average, verify_p):
            ax1.plot(x, i_val, label=f"p_verify={b}")
            ax2.plot(x, r_val, label=f"p_verify={b}")
        ax2.legend(loc="upper left")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("number of infected entities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("number of recovered entities")
        plt.show()
        return verify_p, verify_vals

    def analyze_propagation(alpha_p, beta_p, verify_p):
        alpha_vals, alpha_res = plot_alpha(alpha_p)
        beta_vals, beta_res = plot_beta(beta_p)
        verify_vals, verify_res = plot_verify(verify_p)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
        ax1.plot(alpha_vals, alpha_res)
        ax2.plot(beta_vals, beta_res)
        ax3.plot(verify_vals, verify_res)
        ax1.set_ylabel("Maximum power consumption")
        ax1.set_xlabel("alpha")
        ax2.set_xlabel("beta")
        ax3.set_xlabel("p_verify")
        plt.show()

    def analyze_acting_params(acting_p, usage_p):
        acting_v, usage_v = [], []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        print("change will act")
        config["sim"]["beta"] = .5
        config["sim"]["alpha"] = .5
        config["sim"]["p_verify"] = .5

        config["sim"]["power_usage"] = 1
        for j in acting_p:
            vals = []
            config["sim"]["p_will_act"] = j
            for s in config["seeds"]:
                config["seed"] = s
                framework = EstimationFramework(config, plot=False)
                x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                    framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
                max_val = max(y_true)
                print(f"max y value: {max_val}")
                vals.append(max_val)

            acting_v.append(mean(vals))

        config["sim"]["p_will_act"] = 1
        for j in acting_p:
            vals = []
            config["sim"]["power_usage"] = j
            for s in config["seeds"]:
                config["seed"] = s
                framework = EstimationFramework(config, plot=False)

                x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                    framework.estimate_power_outage(start, action_start=action_start, y_max=1000)
                max_val = max(y_true)
                print(f"max y value: {max_val}")
                vals.append(max_val)

            usage_v.append(mean(vals))

        ax1.plot(acting_p, acting_v)
        ax2.plot(usage_p, usage_v)

        ax1.set_xlabel("p_will_act")
        ax1.set_ylabel("Maximum power consumption")
        ax2.set_xlabel("power_usage")
        plt.show()

    basic_plot(config, start, action_start, y)
    analyze_propagation([0, 0.2, 0.4, 0.6, 0.8, 0.99], [0, 0.2, 0.4, 0.6, 0.8, 1],
                        [0, 0.2, 0.4, 0.6, 0.8, 1])
    analyze_acting_params(acting_p=[0.2, 0.4, 0.6, 0.8, 1], usage_p=[0.2, 0.4, 0.6, 0.8, 1])


def scenario2():
    start = None
    with open("./config/conspiracy.json", "r") as f:
        config = json.load(f)

    action_start = datetime(2013, 11, 7, 17, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    basic_plot(config, start, action_start)


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

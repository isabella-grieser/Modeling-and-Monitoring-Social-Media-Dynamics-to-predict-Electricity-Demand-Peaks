from data.social_media.process_sqlite import *
from framework import EstimationFramework
from framework import running_mean
import matplotlib.pyplot as plt
import matplotlib.dates as md
from frameworkrun import create_plot
from statistics import mean
import numpy as np

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
    else:
        s_max, s_min, s_average = s_max[start_index:], s_min[start_index:], s_average[start_index:]
        i_max, i_min, i_average = i_max[start_index:], i_min[start_index:], i_average[start_index:]
        r_max, r_min, r_average = r_max[start_index:], r_min[start_index:], r_average[start_index:]

    # ax2.fill_between(x_all, s_min, s_max, color='green', alpha=.5, linewidth=0)
    ax2.plot(x_all, s_average, linewidth=2, color='green', ls=ls, label="susceptible")
    # ax2.fill_between(x_all, i_min, i_max, color='red', alpha=.5, linewidth=0)
    ax2.plot(x_all, i_average, linewidth=2, color='red', ls=ls, label="infected")
    # ax2.fill_between(x_all, r_min, r_max, color='blue', alpha=.5, linewidth=0)
    ax2.plot(x_all, r_average, linewidth=2, color='blue', ls=ls, label="removed")

    ax1.plot(x_all, y_ref, linewidth=2, color='black', label="ref")

if __name__ == "__main__":
    data = get_groenwald()

    start = datetime(2022, 8, 4, 13, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)
    action_start = datetime(2022, 8, 4, 19, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    st, e = 0, -1
    values = data.groupby(pd.Grouper(key="date", freq="30min"))["tweet"].count()[st:]

    window = 3

    values_filtered = running_mean(values.values, window)
    index = values.index[int(window/2):-int(window/2)]

    with open("./config/demand-response.json", "r") as f:
        config = json.load(f)

    x_all, y_ref = None, None
    framework = None
    p_vals = [3, 5, 11, len(values_filtered)]
    labels = ["4:00", "5:00", "8:00", "full dataset"]
    dots = ['--', '-.', 'dotted', '-']
    colors = ['gold', 'lawngreen', 'darkolivegreen', 'blue']
    y_vals, s_vals, i_vals, r_vals = [], [], [], []
    """
    fig, ax1, ax2 = create_plot()
    for p, ls, c, lab in zip(p_vals, dots, colors, labels):
        values_pred = values_filtered[:p]
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, year=2022, plot=False)

            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, iterations=100,
                                                y_max=1000, data=values_pred, minutes=30)
            y_vals.append(y_true)
            s_vals.append(s_true)
            i_vals.append(i_true)
            r_vals.append(r_true)
        plot_basic_timeline(ax1, ax2, x_all, y_vals=y_vals, y_ref=y_ref, label=lab,
                            s_true=s_vals, i_true=i_vals, r_true=r_vals, ls=ls, color=c,
                            start_index=80, end_index=-30)

    ax1.axhline(framework.threshold, color='red', label="threshold")

    ax1.axvline(x=start, color='green', ls=':', label='spread start')
    ax1.axvline(x=action_start, color='red', ls=':', label='action start')

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    handles, labels = ax2.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax2.legend(handles, labels)

    handles, labels = ax1.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax1.legend(handles, labels)

    plt.show()

    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig('images/power_demand.pdf', bbox_inches=extent.expanded(1.2, 1.2))
    fig.savefig('images/sir.pdf', bbox_inches=extent.expanded(1.2, 1.2))
    """
    # model prediction framework
    datapoints = range(6, int(len(values_filtered)), 6)
    y_max = []
    config["sim"]["p_verify"] = 0.03919758446621254
    config["sim"]["alpha"] = 0.172974080900178
    config["sim"]["beta"] = 0.99
    config["network"]["edges"] = int(0.12619829266035892 * config["network"]["nodes"])
    for p in datapoints:
        values_pred = values_filtered[:p]
        framework = EstimationFramework(config, year=2022, plot=False)

        x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
            framework.estimate_power_outage(start, action_start=action_start, iterations=100,
                                            y_max=1000, data=values_pred, minutes=30)
        y_vals.append(y_true)
        diffs = [y_true[i] - y_ref[i] for i in range(len(y_true))]
        y_max.append(y_true[np.argmax(diffs)])

    plt.plot(datapoints, y_max)
    plt.xlabel("Number of datapoints")
    plt.ylabel("Maximum estimated power consumption in kW")
    plt.axhline(framework.threshold, color='red', label="threshold")
    plt.show()

    # plot different adoption rates for evs
    ev_adoption = [0.05, 0.1, 0.25, 0.4]

    power_vals = []
    for p in ev_adoption:
        values_pred = values_filtered
        config["model_args"]["electric_car"]["p"] = p
        vals = []
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, year=2022, plot=False)

            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, iterations=100,
                                                y_max=1000, data=values_pred, minutes=30)
            y_vals.append(y_true)

        average_val = []
        for i in range(len(vals[0])):
            average_val.append(mean(vals[j][i] for j in range(len(vals))))
        power_vals.append(average_val)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    start_i, end_i = 80, -30
    x = x_all[start_i:end_i]

    dots = ['--', '-.', 'dotted', '-']
    colors = ['gold', 'lawngreen', 'darkolivegreen', 'blue']

    for p, v, d in zip(ev_adoption, power_vals, dots):
        ax.plot(x, v[start_i:end_i], label=f"p={p}", ls=d)
    ax.plot(x, y_ref[start_i:end_i], color="black", label=f"ref power consumption")
    ax.set_ylabel("Power consumption in kW")
    ax.set_xlabel("Time")
    ax.axhline(framework.threshold, color='red', label="power threshold")
    ax.legend(loc="upper right")
    xfmt = md.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45)
    plt.show()
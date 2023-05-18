import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime


def plot_from_date(x, y_true, y_ref, start_date, end_date, spread_start=None, power_start=None, y_max=1000, si="MW"):
    x_np = np.asarray(x, dtype="datetime64[ns]")
    start_index, = np.where(np.in1d(x_np,[np.datetime64(start_date)]))
    end_index, = np.where(np.in1d(x_np, [np.datetime64(end_date)]))

    start_index = start_index[0]
    end_index = end_index[0]
    x_plot, y_true_plot, y_ref_plot = x[start_index:end_index], y_true[start_index:end_index], y_ref[start_index:end_index]
    fig, ax = plt.subplots()

    xfmt = md.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(xfmt)
    # plt.gca().xaxis.set_major_locator(md.HourLocator(interval=12))
    plt.gcf().autofmt_xdate()
    ax.set_ylabel(f"Power Consumption in {si}")
    ax.set_xlabel(f"Time")

    ax.plot(x_plot, y_true_plot, lw=2, color='blue', label="true consumption")
    ax.plot(x_plot, y_ref_plot, lw=2, color='red', label="ref consumption")
    if spread_start is not None:
        ax.plot([spread_start, spread_start], [0, y_max], lw=2, color="yellow", ls=':', label="spread")
    if power_start is not None:
        ax.plot([power_start, power_start], [0, y_max], lw=2, color="red", ls=':', label="action")


    ax.legend()
    plt.show()


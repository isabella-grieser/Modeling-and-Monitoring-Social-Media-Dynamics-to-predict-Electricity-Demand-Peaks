import matplotlib
from data.social_media.process_sqlite import *
from scipy import signal
from sim.parameterEstimator import *
import matplotlib.pyplot as plt
import matplotlib.dates as md

if __name__ == "__main__":
    data = get_typhoon_data()

    values = data.groupby(pd.Grouper(key="date", freq="15min"))["tweet"].count()

    values = values
    start = values.index[0]
    y = signal.savgol_filter(values.values, 24, 2)

    # estimate parameters
    start_time, end_time, time_step = 0, len(y), 1
    i, r, s = values.values[0], 0, values.values.max() * 0.8
    beta, alpha, p_verify, degree = 0.1, 0.4, 0.4, 10

    return_dict = solve_params(s, i, r, start_time, end_time, time_step, values.values, beta, alpha, degree,
                               p_verify)

    n = return_dict["s_init"] + return_dict["i_init"] + return_dict["r_init"]
    # find the ratio of nodes and edge degree
    ratio = return_dict["degree"] / n
    print(f"estimated params: p_verify: {return_dict['p_verify']}, "
          f"alpha: {return_dict['alpha']}, beta: {return_dict['beta']}, degree ratio: {ratio}")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(values.index, values, label='dataset')
    ax.plot(values.index, return_dict["s"], label='susceptible')
    ax.plot(values.index, return_dict["i"], label='infected')
    ax.plot(values.index, return_dict["r"], label='removed')

    xfmt = md.DateFormatter('%d.%m.')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_ylabel("Number of posts")
    ax.set_xlabel("Time of post")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.show()


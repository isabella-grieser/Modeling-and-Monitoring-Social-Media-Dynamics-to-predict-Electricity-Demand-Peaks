import matplotlib
from data.social_media.process_sqlite import *
from scipy import signal
from sim.parameterEstimator import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = get_typhoon_data()

    values = data.groupby(pd.Grouper(key="date", freq="15min"))["tweet"].count()

    values = values[93:]
    start = values.index[93]
    y = signal.savgol_filter(values.values, 53, 3)

    # estimate parameters
    start_time, end_time, time_step = 0, len(y), 1
    i, r, s = values.values[0], 0, values.values.max() * 0.8
    beta, alpha, p_verify, degree = 0.1, 0.4, 0.4, 10

    return_dict = solve_params(s, i, r, start_time, end_time, time_step, values.values, beta, alpha, degree,
                               p_verify)

    plt.plot(values.index, values, label='dataset')
    plt.plot(values.index, return_dict["s"], label='susceptible')
    plt.plot(values.index, return_dict["i"], label='infected')
    plt.plot(values.index, return_dict["r"], label='removed')

    plt.xticks(rotation='vertical')
    plt.legend(loc="upper right")
    plt.xlabel("Time of post")
    plt.ylabel("Number of posts")

    plt.show()

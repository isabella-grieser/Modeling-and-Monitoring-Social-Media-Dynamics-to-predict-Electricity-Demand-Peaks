import matplotlib.pyplot as plt
from data.social_media.process_sqlite import *
from sim.parameterEstimator import *
from scipy import signal

if __name__ == "__main__":

    data = get_typhoon_data()

    values = data.groupby(pd.Grouper(key="date", freq="15min"))["tweet"].count()

    values = values[93:]
    y = signal.savgol_filter(values.values, 53, 3)

    plt.plot(values.index, y, label='true function')
    plt.xticks(rotation='vertical')
    # try out the parameter estimation algorithm
    i = values.values[0]
    r = 0
    s = values.max() * 0.8
    start_time = 0
    end_time = len(values)
    time_step = 1
    twitter_data = values.values
    beta = 0.1
    alpha = 0.4
    p_verify = 0.4
    degree = 10

    return_dict = solve_params(s, i, r, start_time, end_time, time_step, twitter_data, beta, alpha, degree, p_verify)

    plt.plot(values.index, return_dict["s"], label='susceptible')
    plt.plot(values.index, return_dict["i"], label='infected')
    plt.plot(values.index, return_dict["r"], label='removed')

    plt.legend(loc="upper right")

    plt.show()
from data.social_media.process_sqlite import *
from sim.parameterEstimator import *
import matplotlib.pyplot as plt
import matplotlib.dates as md
from utils.utils import *

if __name__ == "__main__":
    data = get_groenwald()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    st, e = 0, -1
    values = data.groupby(pd.Grouper(key="date", freq="30min"))["tweet"].count()[st:e]
    start = values.index[0]

    window = 3

    values_filtered = running_mean(values.values, window)
    index = values.index[int(window/2):-int(window/2)]

    full_vals = values_filtered
    # estimate parameters
    start_time, end_time, time_step = 0, len(full_vals), 1
    i, r, s = full_vals[0], 0, full_vals.max() * 0.8
    beta, alpha, p_verify, degree_ratio = 0.1, 0.4, 0.4, 0.25


    return_dict = solve_params(s, i, r, start_time, time_step, full_vals, beta, alpha, degree_ratio,
                               p_verify, end_time2=len(index))

    n = return_dict["s_init"] + return_dict["i_init"] + return_dict["r_init"]
    # find the ratio of nodes and edge degree
    print(f"estimated params: p_verify: {return_dict['p_verify']}, "
          f"alpha: {return_dict['alpha']}, beta: {return_dict['beta']}")

    rang = index
    ax.plot(rang, values_filtered, label='dataset', color='steelblue')
    #ax.plot(rang, return_dict["s"], label='s', color='green')
    ax.plot(rang, return_dict["i"], label='full estimation', color='red')
    #ax.plot(rang, return_dict["r"], label='r', color='blue')

    p_vals = [4, 8, 12, 16]
    n_vals = []
    def define_time_str(p):
        hours, mins = divmod(p * 30, 60)
        if mins == 0:
            return f"{hours}h"
        return f"{hours}:{mins}h"

    for p, c in zip(p_vals, ['lawngreen', 'mediumspringgreen', "green", 'gold', 'yellow']):
        values_pred = values_filtered[:p]
        # estimate parameters
        start_time_pred, end_time_pred, time_step_pred = 0, len(values_pred), 1
        i, r, s = values_pred[0], 0, values_pred.max() * 0.8

        return_dict_pred = solve_params(s, i, r, start_time_pred, time_step_pred, values_pred,
                                        beta, alpha, degree_ratio, p_verify, end_time2=len(full_vals))

        print(f"estimated param prediction for {define_time_str(p)}: p_verify: {return_dict_pred['p_verify']}, "
              f"alpha: {return_dict_pred['alpha']}, beta: {return_dict_pred['beta']}")
        infects = return_dict_pred["i"]
        print(f"start slope for {define_time_str(p)}: {(infects[1] - infects[0])/0.5} between {index[0]} and {index[1]}")
        n_vals.append(return_dict_pred['s_init'] + return_dict_pred['i_init'] + return_dict_pred['r_init'])
        range_1 = index[:p]
        range_2 = index[p:len(values_filtered)]
        #ax.plot(range_1, return_dict_pred["s"][:p], label='predicted s', color='lime')
        ax.plot(range_1, return_dict_pred["i"][:p], label=f'i_pred after {define_time_str(p)}', color=c)
        #ax.plot(range_1, return_dict_pred["r"][:p], label='predicted r', color='cornflowerblue')

        #ax.plot(range_2, return_dict_pred["s"][p:len(values_filtered)], color='lime', linestyle='--')
        ax.plot(range_2, return_dict_pred["i"][p:len(values_filtered)], linestyle='--', color=c)
        #ax.plot(range_2, return_dict_pred["r"][p:len(values_filtered)], color='cornflowerblue', linestyle='--')

    ax.set_ylim([0, 100])

    xfmt = md.DateFormatter('%H:%M')
    #xfmt = md.DateFormatter('%d.%m.')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_ylabel("Number of posts")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)

    # ax2 = fig.add_subplot(122)

    # ax2.plot([define_time_str(p) for p in p_vals], n_vals)
    # ax2.set_ylabel("Total number of estimated entities")
    # ax2.set_xlabel("Prediction time")

    plt.show()


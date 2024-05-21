from data.social_media.process_sqlite import *
#from sim.parameterEstimator import *
from sim.diffParamEstimator import *
import matplotlib.pyplot as plt
import matplotlib.dates as md
from utils.utils import *

if __name__ == "__main__":
    np.random.seed(seed=42)
    data = get_groenwald()

    fig, (ax) = plt.subplots(1, 1, figsize=(7,4))
    # fig.subplots_adjust(bottom=0.05)

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
    beta, alpha, p_verify = 0.1, 0.4, 0.4

    degree_ratio = 25

    #return_dict = solve_params(s, i, r, start_time, time_step, full_vals, beta, alpha, degree_ratio,
    #                          p_verify, end_time2=len(index))
    return_dict = solve_params(full_vals, beta, alpha, p_verify, 1000, degree_ratio, pred_iterations=len(index))

    n = return_dict["n"]
    # find the ratio of nodes and edge degree
    print(f"estimated params: p_verify: {return_dict['p_verify']}, "
          f"alpha: {return_dict['alpha']}, beta: {return_dict['beta']}, "
          f"beta_norm: {return_dict['beta'] * degree_ratio / return_dict['n']}, "
          f"N: {return_dict['n']}")

    rang = index
    ax.plot(rang, values_filtered, color='steelblue', label='number of posts every 30 minutes')
    #ax.plot(rang, return_dict["s"], label='estimated S', color='green')
    ax.plot(rang, return_dict["i"], label='est. infection propagation', color='red')
    #ax.plot(rang, return_dict["r"], label='estimated R', color='blue')

    ax.set_ylim([0, 70])

    xfmt = md.DateFormatter('%H:%M')
    #xfmt = md.DateFormatter('%d.%m.')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_ylabel("Number of posts")
    ax.set_xlabel("Time")
    ax.margins(x=0)
    ax.legend(loc="upper right")
    # plt.xticks(rotation=45)

    # ax2 = fig.add_subplot(122)

    # ax2.plot([define_time_str(p) for p in p_vals], n_vals)
    # ax2.set_ylabel("Total number of estimated entities")
    # ax2.set_xlabel("Prediction time")

    plt.show()
    #plt.savefig("images/predictions.pdf",bbox_inches='tight', format="pdf")

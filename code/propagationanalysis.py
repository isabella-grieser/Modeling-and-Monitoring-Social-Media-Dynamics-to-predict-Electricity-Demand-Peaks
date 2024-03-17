import matplotlib.pyplot as plt
from framework import EstimationFramework
from statistics import mean
from data.social_media.process_sqlite import *
import matplotlib.dates as md
from utils.utils import running_mean

colors = ["blue", "red", "green"]
def plot_vals(config, attr1, attr2, probs, start, action_start, iterations=200, year=2022):
    final_vals = []
    i_average = []
    max_i, min_i = [], []
    print(f"change {attr2}")
    x = None
    for j in probs:
        vals = []
        all_i = []
        config[attr1][attr2] = j
        for s in config["seeds"]:
            config["seed"] = s
            framework = EstimationFramework(config, year=year, plot=False)
            x_start, x_all, y_true, y_ref, s_true, i_true, r_true = \
                framework.estimate_power_outage(start, action_start=action_start, iterations=iterations, y_max=1000)
            max_val = max(y_true)
            x = x_all
            print(f"max y value: {max_val}")
            vals.append(max_val)
            all_i.append(i_true)



        average_i, maximum_i, minimum_i = [], [], []
        for i in range(len(all_i[0])):
            average_i.append(mean(all_i[j][i] for j in range(len(all_i))))
            maximum_i.append(max(all_i[j][i] for j in range(len(all_i))))
            minimum_i.append(min(all_i[j][i] for j in range(len(all_i))))
        i_average.append(average_i)
        max_i.append(maximum_i)
        min_i.append(minimum_i)
    return x, i_average, max_i, min_i

def plot_alpha(alpha_p, config, start, action_start):
    x, i_average, max_i, min_i = plot_vals(config, "sim",
                                                    "alpha", alpha_p, start, action_start)

    fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))
    fig.subplots_adjust(bottom=0.05)

    start_i, end_i = 80, -150
    x = x[start_i: end_i]
    for maxi, mini, i_val, b, c in zip(max_i, min_i, i_average, alpha_p, colors):
        ax1.plot(x, i_val[start_i: end_i], label=f"alpha={b}", color=c)
        ax1.fill_between(x, maxi[start_i: end_i], mini[start_i: end_i], alpha=.3, linewidth=0, color=c)
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of infected entities")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    xfmt = md.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(xfmt)
    plt.show()

    plt.savefig("images/alpha.pdf",bbox_inches='tight', format="pdf")
def plot_beta(beta_p, config, start, action_start):
    x, i_average, max_i, min_i = plot_vals(config, "sim",
                                                   "beta", beta_p, start, action_start)

    fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))
    fig.subplots_adjust(bottom=0.05)

    start_i, end_i = 80, -150
    x = x[start_i: end_i]
    for maxi, mini, i_val, b, c in zip(max_i, min_i, i_average, beta_p, colors):
        ax1.plot(x, i_val[start_i: end_i], label=f"beta={b}", color=c)
        ax1.fill_between(x, maxi[start_i: end_i], mini[start_i: end_i], alpha=.3, linewidth=0, color=c)
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of infected entities")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax1.tick_params(labelrotation=45)
    xfmt = md.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(xfmt)
    plt.show()
    plt.savefig("images/beta.pdf",bbox_inches='tight', format="pdf")

def plot_verify(verify_p, config, start, action_start):
    x, i_average, max_i, min_i = plot_vals(config, "sim",
                                                     "p_verify", verify_p, start, action_start)

    fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))
    fig.subplots_adjust(bottom=0.05)

    start_i, end_i = 80, -150
    x = x[start_i: end_i]
    for maxi, mini, i_val, b, c in zip(max_i, min_i, i_average, verify_p, colors):
        ax1.plot(x, i_val[start_i: end_i], label=f"p_verify={b}", color=c)
        ax1.fill_between(x, maxi[start_i: end_i], mini[start_i: end_i], alpha=.3, linewidth=0, color=c)
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of infected entities")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax1.tick_params(labelrotation=45)
    xfmt = md.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(xfmt)
    plt.show()
    plt.savefig("images/verify.pdf",bbox_inches='tight', format="pdf")

def analyze_propagation(alpha_p, beta_p, verify_p, config, start, action_start):
    config["sim"]["beta"] = .2
    config["sim"]["alpha"] = .4
    config["sim"]["p_verify"] = .2
    # plot_alpha(alpha_p, config, start, action_start)
    config["sim"]["alpha"] = .4
    plot_beta(beta_p, config, start, action_start)
    config["sim"]["beta"] = .2
    plot_verify(verify_p, config, start, action_start)

if __name__ == "__main__":

    start = datetime(2022, 11, 20, 16, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)
    action_start = datetime(2022, 11, 20, 19, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)

    with open("./config/demand-response.json", "r") as f:
        config = json.load(f)

    analyze_propagation([0.1, 0.5, 0.9], [0.1, 0.5, 0.9],
                        [0, 0.1, 0.25, 0.4], config, start, start)
import config.systemconstants as const
import numpy as np

def sum_demand(G):
    return sum(G.nodes[n][const.POWER_USAGE] for n in G.nodes)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
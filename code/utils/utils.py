import config.systemconstants as const


def sum_demand(G):
    return sum(G.nodes[n][const.POWER_USAGE_ATTR] for n in G.nodes)

import networkx as nx
import config.systemconstants as const


def create_social_network_graph(n, type, args):
    if type == "watts_strogatz":
        return initialize_node_attributes(nx.watts_strogatz_graph(n, args["edges"], args["p"], args["seed"]))
    if type == "barabasi_albert":
        return initialize_node_attributes(nx.barabasi_albert_graph(n, args["edges"], args["seed"]))
    if type == "erdos_renyi":
        return initialize_node_attributes(nx.erdos_renyi_graph(n, args["p"], args["seed"]))
    if type == "newmann_watts":
        return initialize_node_attributes(nx.newman_watts_strogatz_graph(n, args["edges"], args["p"], args["seed"]))


def initialize_node_attributes(G):
    for n in G.nodes():
        G.nodes[n][const.INFECTION_STATUS] = const.InfectionStatus.SUSCEPTIBLE
        G.nodes[n][const.POWER_USAGE] = 0.0
    return G

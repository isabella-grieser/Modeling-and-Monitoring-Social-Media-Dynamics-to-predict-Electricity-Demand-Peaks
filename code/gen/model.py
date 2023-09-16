import random

import networkx as nx
import config.systemconstants as const


def create_social_network_graph(n, type, args):
    if type == "watts_strogatz":
        return initialize_node_attributes(nx.watts_strogatz_graph(n, args["edges"], args["p"], args["seed"]))
    if type == "barabasi_albert":
        return initialize_node_attributes(nx.barabasi_albert_graph(n, args["edges"], args["seed"]))
    if type == "erdos_renyi":
        return initialize_node_attributes(nx.erdos_renyi_graph(n, args["p"], args["seed"]))

def initialize_node_attributes(G):
    for n in G.nodes():
        G.nodes[n][const.INFECTION_STATUS] = const.InfectionStatus.SUSCEPTIBLE
        G.nodes[n][const.POWER_USAGE] = 0.0
        G.nodes[n][const.ACTIVATED] = False
        G.nodes[n][const.HOUSEHOLD_APPLIANCE] = []
        G.nodes[n][const.WILL_ACT] = False
        G.nodes[n][const.CAN_ACTIVATE] = True
        G.nodes[n][const.PREV_STATE] = const.InfectionStatus.SUSCEPTIBLE
        G.nodes[n][const.P_S] = 0.0
        G.nodes[n][const.P_I] = 0.0
        G.nodes[n][const.P_R] = 0.0
    return G


def define_appliance_use(G, appliances):
    for n in G.nodes():
        for appliance in appliances:
            if random.random() < appliances[appliance]["p"]:
                vals = [appliances[appliance]["power"], appliances[appliance]["duration"]]
                G.nodes[n][const.HOUSEHOLD_APPLIANCE].append(vals)
    return G

def define_availability(G, config):
    p = config["available"]
    for n in G.nodes():
        G.nodes[n][const.CAN_ACTIVATE] = random.random() < p
    return G
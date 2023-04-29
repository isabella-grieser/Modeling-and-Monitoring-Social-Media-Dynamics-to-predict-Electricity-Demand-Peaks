import networkx as nx
import random
import numpy as np


def create_social_network_graph(n, type, args):
    if type == "watts_strogatz":
        return nx.watts_strogatz_graph(n, args["edges"], args["p"], args["seed"])
    if type == "barabasi_albert":
        return nx.barabasi_albert_graph(n, args["edges"], args["seed"])
    if type == "erdos_renyi":
        return nx.erdos_renyi_graph(n, args["p"], args["seed"])


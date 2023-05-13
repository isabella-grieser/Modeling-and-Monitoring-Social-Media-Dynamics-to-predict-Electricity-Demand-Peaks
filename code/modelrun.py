import json
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import gen.model as mb
import networkx as nx
from datetime import datetime
import gen.model as mb
from sim.simulator import Simulator


if __name__ == "__main__":
    with open("./config/network_gen.json", "r") as f:
        config = json.load(f)

    # create social media network model

    # small world novel
    network_model = mb.create_social_network_graph(
        20,
        "watts_strogatz",
        config["network"]
    )

    path = "data/time_15min.csv"
    df = pd.read_csv(path)

    simulator = Simulator(network_model,
                          pd.to_datetime(df["cet_cest_timestamp"], utc=True),
                          df["DE_load_actual_entsoe_transparency"],
                          reduce_factor=100,
                          si="MW")

    simulator.iterate(1000, plot=True, save=False)

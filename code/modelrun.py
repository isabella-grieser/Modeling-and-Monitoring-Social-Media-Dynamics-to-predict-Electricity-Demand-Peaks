import json
import pandas as pd
from datetime import datetime
import datetime as dt
import matplotlib
import gen.model as mb
import utils.plots as plots
import networkx as nx
import gen.model as mb
from sim.simulator import Simulator
import sys

matplotlib.use("TkAgg")

if __name__ == "__main__":
    with open("./config/sim_args.json", "r") as f:
        config = json.load(f)

    # create social media network model

    # small world novel
    network_model = mb.create_social_network_graph(
        20,
        "watts_strogatz",
        config["network"]
    )

    path = "data/household_15min.csv"
    df = pd.read_csv(path)

    start_spread = datetime(2019, 1, 2, 2, 30, 0, tzinfo=dt.timezone.utc)
    start_power = datetime(2019, 1, 2, 4, 30, 0, tzinfo=dt.timezone.utc)

    zone = 'Europe/Berlin'
    x = pd.to_datetime(df["index"], utc=True).dt.to_pydatetime().tolist()
    y = []
    # households considered in sim
    households = ['HH3', 'HH4', 'HH5', 'HH7', 'HH8', 'HH9', 'HH10', 'HH11', 'HH12', 'HH14',
                  'HH16', 'HH18', 'HH19', 'HH20', 'HH21', 'HH22', 'HH23', 'HH27', 'HH28',
                  'HH29', 'HH30', 'HH32', 'HH35', 'HH36', 'HH38', 'HH39']
    for h in households:
        y.append(df[h].to_list())

    simulator = Simulator(network_model,
                          x,
                          y,
                          spread_start=start_spread,
                          power_start=start_power,
                          days=1,
                          args=config,
                          y_max=1000,
                          reduce_factor=100,
                          si="MW"
                          )

    # simulator.iterate(1000, plot=True, save=False, intervall_time=50)
    x_all, y_true, y_ref = simulator.iterate(1000)

    start_date = datetime(2019, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    end_date = datetime(2019, 1, 2, 10, 0, 0, tzinfo=dt.timezone.utc)

    plots.plot_from_date(x, y_true, y_ref, start_date, end_date,
                         spread_start=start_spread, power_start=start_power, y_max=300)


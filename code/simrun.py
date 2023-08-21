import json
import pandas as pd
from datetime import datetime
import datetime as dt
import matplotlib
import gen.model as mb
from sim.simulator import Simulator
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

if __name__ == "__main__":
    with open("./config/sim_args.json", "r") as f:
        config = json.load(f)

    # create social media network model

    network_model = mb.create_social_network_graph(
        config["network"]["nodes"],
        "watts_strogatz",
        config["network"]
    )
    # https://www.destatis.de/EN/Themes/Society-Environment/Income-Consumption-Living-Conditions/Equipment-Consumer-Durables/Tables/equipment-household-appliances-lwr-d.html
    network_model = mb.define_appliance_use(network_model, config["model_args"])
    path = "data/energy/lastprofil_h0i_2023.xls"
    start_index = 1160
    df = pd.read_excel(path, header=None, names=["time", "power"])

    plt.plot(df.time[:350], df.power[:350])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.show()

    start_spread = datetime(2023, 1, 14, 8, 30, 0, tzinfo=dt.timezone.utc)
    start_power = datetime(2023, 1, 14, 10, 30, 0, tzinfo=dt.timezone.utc)

    x = pd.to_datetime(df["time"], utc=True).dt.to_pydatetime().tolist()[start_index:]
    y = [df["power"].to_list()[start_index:]]

    simulator = Simulator(network_model,
                          x,
                          y,
                          spread_start=start_spread,
                          power_start=start_power,
                          days=1,
                          args=config["sim"],
                          seed=config["seed"],
                          y_max=150000,
                          reduce_factor=1,
                          si="kW"
                          )

    simulator.iterate(200, plot=True, save=False, intervall_time=50, draw_graph=True)
    #x_all, y_true, y_ref = simulator.iterate(1000)

    start_date = datetime(2023, 1, 14, 3, 30, 0, tzinfo=dt.timezone.utc)
    end_date = datetime(2023, 1, 14, 14, 30, 0, tzinfo=dt.timezone.utc)

    #plots.plot_from_date(x, y_true, y_ref, start_date, end_date,
    #                     spread_start=start_spread, power_start=start_power, y_max=100000)

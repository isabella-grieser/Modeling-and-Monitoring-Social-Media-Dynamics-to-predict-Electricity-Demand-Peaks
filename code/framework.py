import json
import pandas as pd
from sim.parameterEstimator import *
from gen.model import *
from sim.simulator import Simulator
from utils.plots import *

class EstimationFramework:

    def __init__(self):
        pass

    def estimate_power_outage(self, twitter_data, start,
                              config_path="./config/framework_config.json",
                              ref_energy_path="data/energy/lastprofil_h0i_2023.xls",
                              days=1, beta=0.1, alpha=0.4,
                              p_verify=0.4, degree=10,
                              y_max=500000, index_shift=100):

        # create simulation framework
        with open(config_path, "r") as f:
            config = json.load(f)

        # estimate parameters
        start_time, end_time, time_step = 0, len(twitter_data), 1
        i, r, s = twitter_data.values[0], 0, twitter_data.max() * 0.8

        return_dict = solve_params(s, i, r, start_time, end_time, time_step, twitter_data, beta, alpha, degree,
                                   p_verify)

        # create dynamic config for model generation
        nodes = config["network"]["n"]
        n = return_dict["s_init"] + return_dict["i_init"] + return_dict["r_init"]

        # find the ratio of nodes and edge degree
        ratio = return_dict["degree"] / n

        model_config = {
            "seed": config["seed"],
            "edges": round(ratio * nodes)
        }

        social_network_model = create_social_network_graph(nodes, "barabasi_albert", model_config)

        config["sim"]["p_verify"] = return_dict["p_verify"]
        config["sim"]["alpha"] = return_dict["alpha"]
        config["sim"]["beta"] = return_dict["beta"]

        social_network_model = define_appliance_use(social_network_model, config["model_args"])
        df = pd.read_excel(ref_energy_path, header=None, names=["time", "power"])
        x = pd.to_datetime(df["time"], utc=True).dt.to_pydatetime().tolist()

        comparable_starts = [x_i[0] for x_i in enumerate(x) if x_i[1].time() > start.time()
                             and x_i[1].day == start.day and x_i[1].month == start.month]
        start_index = comparable_starts[0] - index_shift
        spread_start = x[comparable_starts[0]]
        y_val = [df["power"].to_list()[start_index:]]
        x_val = x[start_index:]

        simulator = Simulator(social_network_model, x_val, y_val, args=config["sim"],
                              seed=config["seed"], spread_start=spread_start, days=days,
                              y_thresh_factor=config["sim"]["power_threshold"], y_max=y_max)

        x_all, y_ref, y_true = simulator.iterate(200, plot=True, save=False, intervall_time=50)

        max_value = simulator.power_thresh
        # find first occurrence where value is true
        val = next(iter([y_i[0] for y_i in enumerate(y_ref) if y_i[1] > max_value]), -1)

        print(val)
        if val == -1:
            return None
        else:
            return x_all[val], x_all, y_true, y_ref

import json

import pytz

from sim.parameterEstimator import *
from gen.model import *
from sim.simulator import Simulator
from demandlib import bdew


class EstimationFramework:

    def __init__(self, config, year=2013, plot=True):
        self.config = config
        self.year = year
        self.plot = plot
        self.threshold = None

    def estimate_power_outage(self, start, action_start=None, data=None,
                              factor=1,
                              days=1, beta=0.1, alpha=0.4,
                              p_verify=0.4, degree=10, iterations=200,
                              y_max=5000, index_shift=100, edge_ratio=0.0065):

        # estimate parameters
        if data is not None:
            start_time, end_time, time_step = 0, len(data), 1
            i, r, s = data[0], 0, data.max() * 0.8

            return_dict = solve_params(s, i, r, start_time, end_time, time_step, data, beta, alpha, degree,
                                       p_verify)

            # create dynamic config for model generation
            nodes = self.config["network"]["nodes"]
            n = return_dict["s_init"] + return_dict["i_init"] + return_dict["r_init"]

            # find the ratio of nodes and edge degree
            ratio = return_dict["degree"] / n

            model_config = {
                "seed": self.config["seed"],
                "edges": round(ratio * nodes)
            }

            social_network_model = create_social_network_graph(nodes, "barabasi_albert", model_config)
            self.config["sim"]["p_verify"] = return_dict["p_verify"]
            self.config["sim"]["alpha"] = return_dict["alpha"]
            self.config["sim"]["beta"] = return_dict["beta"]

            print(f"estimated params: p_verify: {return_dict['p_verify']}, "
                f"alpha: {return_dict['alpha']}, beta: {return_dict['beta']}, degree ratio: {ratio}")
        else:
            model_config = {
                "seed": self.config["seed"],
                "edges": round(edge_ratio * self.config["network"]["nodes"])
            }
            social_network_model = create_social_network_graph(self.config["network"]["nodes"],
                                                               "barabasi_albert", model_config)

        social_network_model = define_appliance_use(social_network_model, self.config["model_args"])
        social_network_model = define_availability(social_network_model, self.config["network"])

        # load profiles are all in kWh
        load_profile = bdew.elec_slp.ElecSlp(self.year)
        df = load_profile.get_profile({'h0': self.year})
        x = df.index.to_list()
        x = [x_i.replace(tzinfo=pytz.UTC) for x_i in x]
        y = df.h0.apply(lambda h: h * factor).to_list()

        comparable_starts = [x_i[0] for x_i in enumerate(x) if x_i[1].time() > start.time()
                             and x_i[1].day == start.day and x_i[1].month == start.month]
        start_index = comparable_starts[0] - index_shift
        spread_start = x[comparable_starts[0]]
        y_val = [y[start_index:]]
        x_val = x[start_index:]

        simulator = Simulator(social_network_model, x_val, y_val, args=self.config["sim"],
                              seed=self.config["seed"], spread_start=spread_start, si="kW",
                              power_start=action_start, days=days, y_max=y_max,
                              nr_init_nodes=1)

        x_all, y_true, y_ref, s_true, i_true, r_true = \
            simulator.iterate(iterations, plot=self.plot, save=False, intervall_time=50)

        self.threshold = simulator.power_thresh
        # find first occurrence where value is true
        index = next(iter([y_i[0] for y_i in enumerate(y_ref) if y_i[1] > self.threshold]), -1)

        return index, x_all, y_true, y_ref, s_true, i_true, r_true

import sys

import numpy as np
import networkx as nx
from random import sample
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import config.systemconstants as const
import utils.utils as util
from matplotlib.lines import Line2D
import datetime
import math

matplotlib.use("TkAgg")


class Simulator:
    def __init__(self, G, x, y, args, spread_start=None,
                 power_start=None,
                 seed=42, days=1,
                 minutes=15,
                 steps=1, si="MW",
                 reduce_factor=1,
                 y_max=50000):
        self.graph = G
        self.x = x
        self.y = y
        self.args = args
        self.spread_start = spread_start
        self.power_start = power_start
        self.seed = seed
        self.steps = steps
        self.y_max = y_max
        self.si = si
        self.reduce_factor = reduce_factor
        self.minutes = minutes
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.timespan = 0
        first_date = x[0]
        for d in x[1:]:
            self.timespan += 1
            if d.hour == first_date.hour and d.minute == first_date.minute:
                break
        self.timespan *= days
        self.power_thresh = 0
        self.__initialize__()
        self.power_thresh *= self.args["power_threshold"]

    def __initialize__(self):
        length = len(self.y)
        vector_sum = np.zeros(len(self.y[0]))
        for n in self.graph.nodes:
            # select random household profile for the node
            self.graph.nodes[n][const.HOUSEHOLD_INDEX] = random.randint(0, length - 1)
            self.graph.nodes[n][const.POWER_USAGE] = self.y[self.graph.nodes[n][const.HOUSEHOLD_INDEX]][0]
            vector_sum += np.array(self.y[self.graph.nodes[n][const.HOUSEHOLD_INDEX]])
        self.power_thresh = np.max(vector_sum)

        # infect a random node
        node = sample(list(self.graph.nodes()), 1)
        self.graph.nodes[node[0]][const.INFECTION_STATUS] = const.InfectionStatus.INFECTED

    def iterate(self, iterations=1000, intervall_time=50, draw_graph=False,
                plot=False, save=False, save_name="./output/video.mp4"):

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            line1, = ax1.plot([], [], lw=2, color='blue', label="true consumption")
            line2, = ax1.plot([], [], lw=2, color='black', label="ref consumption")
            power_thresh, = ax1.plot([], [], lw=2, color="red", label="power threshold")
            if self.spread_start is not None:
                spread_starting, = ax1.plot([], [], lw=2, color="yellow", ls=':', label="spread")
            if self.power_start is not None:
                action_starting, = ax1.plot([], [], lw=2, color="red", ls=':', label="action")

            xfmt = md.DateFormatter('%H:%M')
            ax1.xaxis.set_major_formatter(xfmt)
            ax1.set_ylabel(f"Power Consumption in {self.si}")
            ax1.set_xlabel(f"Time")
            ax1.legend(loc="upper left")

            if draw_graph:
                l1 = Line2D([], [], color="black", linestyle='none', marker='o', markerfacecolor="white",
                            label='susceptible')
                l2 = Line2D([], [], color="black", linestyle='none', marker='o', markerfacecolor="red",
                            label='infected')
                l3 = Line2D([], [], color="black", linestyle='none', marker='o', markerfacecolor="grey",
                            label='recovered')
                ax2.set_title("Propagation of misinformation")
                ax2.legend(handles=[l1, l2, l3])

                pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")
            else:
                # add system view, not graph view
                line_sir_s, = ax2.plot([], [], lw=2, color='green', label="susceptible")
                line_sir_i, = ax2.plot([], [], lw=2, color='red', label="infected")
                line_sir_r, = ax2.plot([], [], lw=2, color='blue', label="recovered")

                n_size = self.graph.number_of_nodes()
                s_true = [n_size for n in range(self.timespan)]
                i_true = [0 for n in range(self.timespan)]
                r_true = [0 for n in range(self.timespan)]
                ax2.set_ylabel(f"Infection Process")
                ax2.set_xlabel(f"Time")
                ax2.set(xlim=(self.x[0], self.x[self.timespan + iterations]), ylim=(0, n_size))
                ax2.legend(loc="upper left")

            x_plot = self.x[:self.timespan]
            y_true, y_ref = self.__calculate_power__(x_plot, [y_s[:self.timespan] for y_s in self.y])

            # when initializing -> y_true == y_ref
            y_true, y_ref = list(y_ref), list(y_ref)
            if isinstance(x_plot, pd.Series):
                x_plot = x_plot.to_numpy().tolist()
            if isinstance(y_true, pd.Series):
                y_true = y_true.to_numpy().tolist()
            if isinstance(y_ref, pd.Series):
                y_ref = y_ref.to_numpy().tolist()

            if isinstance(y_ref, pd.Series):
                y_ref = y_ref.to_numpy().tolist()

            # if it is necessary to use the ref data multiple times
            times = 0

            def update_graph():
                s_nodes = [n_n for n_n, y in self.graph.nodes(data=True) if
                           y[const.INFECTION_STATUS] == const.InfectionStatus.SUSCEPTIBLE]
                i_nodes = [n_n for n_n, y in self.graph.nodes(data=True) if
                           y[const.INFECTION_STATUS] == const.InfectionStatus.INFECTED]
                r_nodes = [n_n for n_n, y in self.graph.nodes(data=True) if
                           y[const.INFECTION_STATUS] == const.InfectionStatus.RECOVERED]
                drawn_s_nodes = nx.draw_networkx_nodes(self.graph,
                                                       pos=pos,
                                                       nodelist=set(s_nodes),
                                                       node_color="white",
                                                       ax=ax2)
                drawn_i_nodes = nx.draw_networkx_nodes(self.graph,
                                                       pos=pos,
                                                       nodelist=set(i_nodes),
                                                       node_color="red",
                                                       ax=ax2)
                drawn_r_nodes = nx.draw_networkx_nodes(self.graph,
                                                       pos=pos,
                                                       nodelist=set(r_nodes),
                                                       node_color="grey",
                                                       ax=ax2)
                drawn_s_nodes.set_edgecolor("black")
                drawn_i_nodes.set_edgecolor("black")
                drawn_r_nodes.set_edgecolor("black")

                return drawn_s_nodes, drawn_i_nodes, drawn_r_nodes

            if draw_graph:
                nx.draw_networkx_edges(self.graph,
                                       pos=pos,
                                       ax=ax2,
                                       edge_color="gray")

                sus_nodes, inf_nodes, rem_nodes = update_graph()

            def plot_init():
                returns = []
                line1.set_data(x_plot, y_true)
                line2.set_data(x_plot, y_ref)
                power_thresh.set_data([x_plot[0], x_plot[-1]], [self.power_thresh, self.power_thresh])
                returns.extend([line1, line2, power_thresh])
                if self.spread_start is not None:
                    spread_starting.set_data([], [])
                    returns.append(spread_starting)
                if self.power_start is not None:
                    action_starting.set_data([], [])
                    returns.append(action_starting)
                if not draw_graph:
                    line_sir_s.set_data(x_plot, s_true)
                    line_sir_i.set_data(x_plot, i_true)
                    line_sir_r.set_data(x_plot, r_true)
                    returns.extend([line_sir_s, line_sir_i, line_sir_r])
                return returns

            # to return values: save old vals
            x_total = x_plot.copy()
            y_ref_total = y_true.copy()
            y_true_total = y_ref.copy()

            def animate(frame):
                x_new, y_new_true, y_new_ref = None, None, None
                # append new steps
                times, n = divmod(frame * self.steps + self.timespan, len(self.x))
                if n + self.steps >= len(self.x):
                    i = (n + self.steps) % len(self.x)
                    x_new = self.x[n:] + self.x[:i]
                    y_new_true, y_new_ref = self.__calculate_power__(x_new, [y_s[n:] + y_s[:i] for y_s in self.y])
                    x_plot_new = [
                        x_vals + len(self.x) * times * datetime.timedelta(minutes=self.minutes)
                        for x_vals in self.x[n:]
                    ]
                    x_plot_new.extend([
                        x_vals + len(self.x) * (times + 1) * datetime.timedelta(minutes=self.minutes)
                        for x_vals in self.x[:i]
                    ])
                else:
                    i = n + self.steps
                    x_new = self.x[n:i]
                    y_new_true, y_new_ref = self.__calculate_power__(x_new, [y_s[n:i] for y_s in self.y])
                    x_plot_new = [
                        x_vals + len(self.x) * times * datetime.timedelta(minutes=self.minutes)
                        for x_vals in self.x[n:i]
                    ]
                # propagate based on if the next x (after step size) fulfills the simulation condition
                self.__propagate__(x_new[-1])

                x_plot.extend(x_plot_new)
                y_true.extend(y_new_true)
                y_ref.extend(y_new_ref)

                x_total.extend(x_plot_new)
                y_ref_total.extend(y_new_true)
                y_true_total.extend(y_new_ref)

                del x_plot[:self.steps]
                del y_true[:self.steps]
                del y_ref[:self.steps]

                x_min, x_max = x_plot[0], x_plot[-1]

                if self.spread_start is not None:
                    spread_starting.set_data([self.spread_start, self.spread_start], [0, self.y_max])
                if self.power_start is not None:
                    action_starting.set_data([self.power_start, self.power_start], [0, self.y_max])

                ax1.set(xlim=(x_min, x_max), ylim=(0, self.y_max))
                power_thresh.set_data([x_min, x_max], [self.power_thresh, self.power_thresh])
                line1.set_data(x_plot, y_true)
                line2.set_data(x_plot, y_ref)
                ax1.legend(loc="upper left")
                ax1.set_title(
                    f" Total Power Consumption {x_min.day}/{x_min.month}/{x_min.year}-{x_max.day}/{x_max.month}/{x_max.year}"
                )
                if draw_graph:
                    sus_nodes, inf_nodes, rem_nodes = update_graph()
                    return [line1, line2, sus_nodes, inf_nodes, rem_nodes]
                else:
                    s_sum = 0
                    i_sum = 0
                    r_sum = 0
                    for n_n, y in self.graph.nodes(data=True):
                        if y[const.INFECTION_STATUS] == const.InfectionStatus.SUSCEPTIBLE:
                            s_sum += 1
                        if y[const.INFECTION_STATUS] == const.InfectionStatus.INFECTED:
                            i_sum += 1
                        if y[const.INFECTION_STATUS] == const.InfectionStatus.RECOVERED:
                            r_sum += 1

                    s_true.append(s_sum)
                    i_true.append(i_sum)
                    r_true.append(r_sum)
                    line_sir_s.set_data(x_total, s_true)
                    line_sir_i.set_data(x_total, i_true)
                    line_sir_r.set_data(x_total, r_true)
                    return [line1, line2, line_sir_s, line_sir_i, line_sir_r]

            anim = animation.FuncAnimation(fig, animate,
                                           init_func=plot_init,
                                           frames=iterations,
                                           interval=intervall_time,
                                           repeat=False)

            if save:
                anim.save(save_name, fps=10)
            else:
                plt.show()
            return x_total, y_ref_total, y_true_total
        else:
            x_all = self.x[:self.timespan]
            y_true, y_ref = self.__calculate_power__(x_all, [y_s[:self.timespan] for y_s in self.y])
            for i in range(iterations):
                x_new, y_new_true, y_new_ref = None, None, None
                n = (i * self.steps + self.timespan) % len(self.x)
                if n + self.steps >= len(self.x):
                    i = (n + self.steps) % len(self.x)
                    x_new = self.x[n:] + self.x[:i]
                    y_new_true, y_new_ref = self.__calculate_power__(x_new, [y_s[n:] + y_s[:i] for y_s in self.y])
                else:
                    i = n + self.steps
                    x_new = self.x[n:i]
                    y_new_true, y_new_ref = self.__calculate_power__(x_new, [y_s[n:i] for y_s in self.y])

                self.__propagate__(x_new[-1])
                x_all.extend(x_new)
                y_true.extend(y_new_true)
                y_ref.extend(y_new_ref)

            return x_all, y_true, y_ref

    def __propagate__(self, x):
        if self.spread_start is not None and x < self.spread_start:
            return

        states = [const.InfectionStatus.SUSCEPTIBLE,
                  const.InfectionStatus.INFECTED,
                  const.InfectionStatus.RECOVERED]

        # after conditions are filled: implement SIR Model
        # given the paper "Fact-checking Effect on Viral Hoaxes:
        # A Model of Misinformation Spread in Social Networks"
        p_verify, alpha, beta = self.args["p_verify"], self.args["alpha"], self.args["beta"]

        def calc_state_probabilities(n):
            if self.graph.nodes[n][const.INFECTION_STATUS] == const.InfectionStatus.RECOVERED:
                # status of node cannot be changed
                self.graph.nodes[n][const.P_S] = 0
                self.graph.nodes[n][const.P_I] = 0
                self.graph.nodes[n][const.P_R] = 1
                return
            n_i = sum(1 for x in self.graph.neighbors(n) if
                      self.graph.nodes[x][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED)
            n_r = sum(1 for x in self.graph.neighbors(n) if
                      self.graph.nodes[x][const.INFECTION_STATUS] == const.InfectionStatus.RECOVERED)
            if n_i == 0:
                f_i = 0
            else:
                f_i = beta * ((n_i * (1 + alpha)) / (n_i * (1 + alpha) + n_r * (1 - alpha)))
            if n_r == 0:
                g_i = 0
            else:
                g_i = beta * ((n_r * (1 - alpha)) / (n_i * (1 + alpha) + n_r * (1 - alpha)))

            if math.isnan(f_i):
                f_i = sys.float_info.max
            if math.isnan(g_i):
                g_i = sys.float_info.max

            s_i_s = int(self.graph.nodes[n][const.INFECTION_STATUS] == const.InfectionStatus.SUSCEPTIBLE)
            s_i_i = int(self.graph.nodes[n][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED)
            s_i_r = int(self.graph.nodes[n][const.INFECTION_STATUS] == const.InfectionStatus.RECOVERED)

            self.graph.nodes[n][const.P_S] = (1 - f_i - g_i) * s_i_s
            self.graph.nodes[n][const.P_I] = f_i * s_i_s + (1 - p_verify) * s_i_i
            self.graph.nodes[n][const.P_R] = g_i * s_i_s + p_verify * s_i_i + s_i_r

            if math.isnan(self.graph.nodes[n][const.P_S]) or math.isnan(self.graph.nodes[n][const.P_I]) \
                    or math.isnan(self.graph.nodes[n][const.P_R]):
                val = True

        def change_state(n):
            if not (self.graph.nodes[n][const.CAN_ACTIVATE] and self.args["fringe"]):
                prev_infect_status = self.graph.nodes[n][const.INFECTION_STATUS]

                self.graph.nodes[n][const.INFECTION_STATUS] = np.random.choice(states,
                                                                               p=[self.graph.nodes[n][const.P_S],
                                                                                  self.graph.nodes[n][const.P_I],
                                                                                  self.graph.nodes[n][const.P_R]
                                                                                  ])

                current_infect_status = self.graph.nodes[n][const.INFECTION_STATUS]

                if prev_infect_status == const.InfectionStatus.SUSCEPTIBLE \
                        and current_infect_status == const.InfectionStatus.INFECTED:
                    self.graph.nodes[n][const.WILL_ACT] = random.random() < self.args["p_will_act"]

                if self.power_start is None or self.power_start is not None and x > self.power_start:
                    # check if the node will start using more power
                    if not self.graph.nodes[n][const.ACTIVATED] \
                            and current_infect_status == const.InfectionStatus.INFECTED:
                        self.graph.nodes[n][const.ACTIVATED] = random.random() < self.args["power_usage"]

        n_jobs = 4
        for n in self.graph.nodes:
            calc_state_probabilities(n)
        for n in self.graph.nodes:
            change_state(n)

    def __calculate_power__(self, x_s, y_lists):
        # algorithm to calculate the new demand for each node
        y_true, y_ref = [], []
        for i, x in enumerate(x_s):
            original_power_usage = 0
            for n in self.graph.nodes:
                ref_power = y_lists[self.graph.nodes[n][const.HOUSEHOLD_INDEX]][i]
                node_val = ref_power * self.__power_consumption_factor__(x, n) \
                           + self.__power_consumption_offset__(x, n)
                self.graph.nodes[n][const.POWER_USAGE] = node_val
                original_power_usage += ref_power
            new_y_true = util.sum_demand(self.graph) / self.reduce_factor
            new_y_ref = original_power_usage / self.reduce_factor
            y_true.append(new_y_true)
            y_ref.append(new_y_ref)
        return y_true, y_ref

    def __power_consumption_factor__(self, x, node):
        if self.power_start is not None and x < self.power_start:
            return 1
        elif self.graph.nodes[node][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED \
                and self.graph.nodes[node][const.ACTIVATED] and self.graph.nodes[node][const.WILL_ACT] \
                and self.graph.nodes[node][const.CAN_ACTIVATE]:
            return self.args["factor"]
        return 1

    def __power_consumption_offset__(self, x, node):
        if self.power_start is not None and x < self.power_start:
            return 0
        if self.graph.nodes[node][const.ACTIVATED] and self.graph.nodes[node][const.WILL_ACT] and \
                self.graph.nodes[node][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED \
                and self.graph.nodes[node][const.CAN_ACTIVATE]:
            total_power = 0
            for appliance in self.graph.nodes[node][const.HOUSEHOLD_APPLIANCE]:
                if appliance[1] >= 0:
                    total_power += appliance[0]
                    appliance[1] -= 1
            return total_power
        return 0

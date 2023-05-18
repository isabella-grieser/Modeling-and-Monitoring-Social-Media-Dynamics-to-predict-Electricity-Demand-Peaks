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

matplotlib.use("TkAgg")

class Simulator:
    def __init__(self, G, x, y, args, spread_start=None,
                 power_start=None,
                 mu=1, sigma=0.005, seed=42, days=2,
                 steps=1, si="MW", reduce_factor=1,
                 y_max=50000):
        self.graph = G
        self.x = x
        self.y = y
        self.args = args
        self.spread_start = spread_start
        self.power_start = power_start
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        self.steps = steps
        self.y_max = y_max
        self.si = si
        self.reduce_factor = reduce_factor
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.timespan = 0
        first_date = x[0]
        for d in x[1:]:
            self.timespan += 1
            if d.hour == first_date.hour and d.minute == first_date.minute:
                break
        self.timespan *= days
        self.__initialize__()

    def __initialize__(self):
        length = len(self.y)
        for n in self.graph.nodes:
            self.graph.nodes[n][const.POWER_USAGE] = np.random.normal(self.mu, self.sigma) * self.y[n][0]
            # select random household profile for the node
            self.graph.nodes[n][const.HOUSEHOLD_INDEX] = random.randint(0, length-1)

        # infect a random node
        node = sample(list(self.graph.nodes()), 1)
        self.graph.nodes[node[0]][const.INFECTION_STATUS] = const.InfectionStatus.INFECTED

    def iterate(self, iterations=1000, intervall_time=50, plot=False, save=False):

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            line1, = ax1.plot([], [], lw=2, color='blue', label="true consumption")
            line2, = ax1.plot([], [], lw=2, color='red', label="ref consumption")
            if self.spread_start is not None:
                spread_starting, = ax1.plot([], [], lw=2, color="yellow", ls=':', label="spread")
            if self.power_start is not None:
                action_starting, = ax1.plot([], [], lw=2, color="red", ls=':', label="action")

            xfmt = md.DateFormatter('%H:%M')
            ax1.xaxis.set_major_formatter(xfmt)
            ax1.set_ylabel(f"Power Consumption in {self.si}")
            ax1.set_xlabel(f"Time")
            ax2.set_title("Propagation of misinformation")

            legend = ax1.legend(loc="upper left")

            pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")

            def update_graph():
                s_nodes = [n_n for n_n, y in self.graph.nodes(data=True) if
                           y[const.INFECTION_STATUS] == const.InfectionStatus.SUSCEPTIBLE]
                i_nodes = [n_n for n_n, y in self.graph.nodes(data=True) if
                           y[const.INFECTION_STATUS] == const.InfectionStatus.INFECTED]
                r_nodes = [n_n for n_n, y in self.graph.nodes(data=True) if
                           y[const.INFECTION_STATUS] == const.InfectionStatus.REMOVED]
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

            nx.draw_networkx_edges(self.graph,
                                   pos=pos,
                                   ax=ax2,
                                   edge_color="gray")

            sus_nodes, inf_nodes, rem_nodes = update_graph()

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

            def plot_init():
                line1.set_data(x_plot, y_true)
                line2.set_data(x_plot, y_ref)
                action_starting.set_data([], [])
                spread_starting.set_data([], [])
                return [line1, line2, action_starting, spread_starting]

            def animate(frame):
                x_new, y_new_true, y_new_ref = None, None, None
                # append new steps
                n = (frame * self.steps + self.timespan) % len(self.x)
                if n + self.steps >= len(self.x):
                    i = (n + self.steps) % len(self.x)
                    x_new = self.x[n:] + self.x[:i]
                    y_new_true, y_new_ref = self.__calculate_power__(x_new, [y_s[n:] + y_s[:i] for y_s in self.y])
                else:
                    i = n + self.steps
                    x_new = self.x[n:i]
                    y_new_true, y_new_ref = self.__calculate_power__(x_new, [y_s[n:i] for y_s in self.y])

                # propagate based on if the next x (after step size) fulfills the simulation condition
                self.__propagate__(x_new[-1])

                x_plot.extend(x_new)
                y_true.extend(y_new_true)
                y_ref.extend(y_new_ref)

                del x_plot[:self.steps]
                del y_true[:self.steps]
                del y_ref[:self.steps]

                x_min, x_max = x_plot[0], x_plot[-1]

                if self.spread_start is not None:
                    spread_starting.set_data([self.spread_start, self.spread_start], [0, self.y_max])
                if self.power_start is not None:
                    action_starting.set_data([self.power_start, self.power_start], [0, self.y_max])

                ax1.set(xlim=(x_min, x_max), ylim=(0, self.y_max))
                line1.set_data(x_plot, y_true)
                line2.set_data(x_plot, y_ref)
                ax1.legend(loc="upper left")
                ax1.set_title(
                    f" Total Power Consumption {x_min.day}/{x_min.month}/{x_min.year}-{x_max.day}/{x_max.month}/{x_max.year}"
                )

                sus_nodes, inf_nodes, rem_nodes = update_graph()

                return [line1, line2, sus_nodes, inf_nodes, rem_nodes]

            anim = animation.FuncAnimation(fig, animate,
                                           init_func=plot_init,
                                           frames=iterations,
                                           interval=intervall_time)

            if save:
                anim.save('simulation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

            plt.show()

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

        # after conditions are filled: implement SIR Model
        beta, gamma, check = self.args["sim"]["beta"], self.args["sim"]["gamma"], self.args["sim"]["check"]
        for n in self.graph.nodes:
            if self.graph.nodes[n][const.INFECTION_STATUS] == const.InfectionStatus.SUSCEPTIBLE:
                for neighbor in self.graph.neighbors(n):
                    if self.graph.nodes[neighbor][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED \
                            and random.random() < beta:
                        self.graph.nodes[n][const.INFECTION_STATUS] = const.InfectionStatus.INFECTED
                        continue
            if self.graph.nodes[n][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED:
                for neighbor in self.graph.neighbors(n):
                    if self.graph.nodes[neighbor][const.INFECTION_STATUS] == const.InfectionStatus.REMOVED \
                            and random.random() < gamma:
                        self.graph.nodes[n][const.INFECTION_STATUS] = const.InfectionStatus.REMOVED
                        continue
                if random.random() < check:
                    self.graph.nodes[n][const.INFECTION_STATUS] = const.InfectionStatus.REMOVED

    def __calculate_power__(self, x_s, y_lists):
        # algorithm to calculate the new demand for each node
        y_true, y_ref = [], []
        for i, x in enumerate(x_s):
            original_power_usage = 0
            for n in self.graph.nodes:
                ref_power = np.random.normal(self.mu, self.sigma) * y_lists[n][i]
                self.graph.nodes[n][const.POWER_USAGE] = ref_power * self.__power_consumption_factor__(x, n) \
                                                              + self.__power_consumption_offset__(x, n)
                original_power_usage += ref_power
            y_true.append(util.sum_demand(self.graph) / self.reduce_factor)
            y_ref.append(original_power_usage / self.reduce_factor)
        return y_true, y_ref

    def __power_consumption_factor__(self, x, node):
        if self.power_start is not None and x < self.power_start:
            return 1
        if self.graph.nodes[node][const.INFECTION_STATUS] == const.InfectionStatus.SUSCEPTIBLE:
            return 1
        elif self.graph.nodes[node][const.INFECTION_STATUS] == const.InfectionStatus.INFECTED:
            return 5
        elif self.graph.nodes[node][const.INFECTION_STATUS] == const.InfectionStatus.REMOVED:
            return 1
        else:
            return 1

    def __power_consumption_offset__(self, x, node):
        if self.power_start is not None and x < self.power_start:
            return 0
        return 0

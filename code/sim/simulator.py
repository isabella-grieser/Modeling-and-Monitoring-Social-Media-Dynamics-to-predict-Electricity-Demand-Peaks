import numpy as np
import networkx as nx
import matplotlib;

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import config.systemconstants as const
import utils.utils as util


class Simulator:
    def __init__(self, G, x, y, mu=1, sigma=0.005, seed=42, days=5, steps=1, si="MW", reduce_factor=1, y_min=0,
                 y_max=50000):
        self.graph = G
        self.x = x
        self.y = y
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        self.steps = steps
        self.y_min = y_min
        self.y_max = y_max
        self.si = si
        self.reduce_factor = reduce_factor
        np.random.seed(self.seed)

        self.timespan = 0
        first_date = x[0]
        for d in x[1:]:
            self.timespan += 1
            if d.hour == first_date.hour and d.minute == first_date.minute:
                break
        self.timespan *= days
        self.__initialize__()

    def __initialize__(self):
        for n in self.graph.nodes:
            self.graph.nodes[n][const.POWER_USAGE_ATTR] = np.random.normal(self.mu, self.sigma) * self.y[0]

    def iterate(self, iterations=1000, plot=False, save=False):

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            line1, = ax1.plot([], [], lw=2, color='b')
            line2, = ax1.plot([], [], lw=2, color='r')
            xfmt = md.DateFormatter('%H:%M')
            ax1.xaxis.set_major_formatter(xfmt)
            ax1.set_ylabel(f"Power Consumption in {self.si}")
            ax1.set_xlabel(f"Time")
            ax2.set_title("Propagation of misinformation")

            pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")

            nx.draw_networkx_edges(self.graph,
                                   pos=pos,
                                   ax=ax2,
                                   edge_color="gray")
            normal_nodes = nx.draw_networkx_nodes(self.graph,
                                                  pos=pos,
                                                  nodelist=set(self.graph.nodes()),
                                                  node_color="white",
                                                  ax=ax2)
            normal_nodes.set_edgecolor("black")

            def plot_init():
                line1.set_data([], [])
                line2.set_data([], [])
                return [line1, line2]

            def animate(frame):
                self.__propagate__()
                x, y_true, y_ref = None, None, None
                n = (frame * self.steps + self.timespan) % len(self.x)
                if n + self.timespan >= len(self.x):
                    i = (n * self.steps + self.timespan) % len(self.x)
                    x = self.x[n:] + self.x[:i]
                    y_true, y_ref = zip(*[self.__calculate_power__(v) for v in self.y[n:] + self.y[:i]])
                else:
                    x = self.x[n:self.timespan + n]
                    y_true, y_ref = zip(*[self.__calculate_power__(v) for v in self.y[n:self.timespan + n]])

                x_min, x_max = x.min(), x.max()
                ax1.set(xlim=(x_min, x_max), ylim=(self.y_min, self.y_max))
                line1.set_data(x, y_true)
                line2.set_data(x, [j + 1000.0 for j in y_ref])
                ax1.set_title(
                    f" Total Power Consumption {x_min.day}/{x_min.month}/{x_min.year}-{x_max.day}/{x_max.month}/{x_max.year}"
                    )

                normal_nodes = nx.draw_networkx_nodes(self.graph,
                                                      pos=pos,
                                                      nodelist=set(self.graph.nodes()),
                                                      node_color="white",
                                                      ax=ax2)
                normal_nodes.set_edgecolor("black")
                return [line1, line2, normal_nodes]

            anim = animation.FuncAnimation(fig, animate,
                                           init_func=plot_init,
                                           frames=iterations,
                                           interval=1)

            if save:
                anim.save('simulation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

            plt.show()

        else:
            y_true, y_ref, x_all = [], [], []
            for i in range(iterations):
                self.__propagate__()
                x, y_1, y_2 = None, None, None
                n = (i * self.steps + self.timespan) % len(self.x)
                if n + self.timespan >= len(self.x):
                    i = (n * self.steps + self.timespan) % len(self.x)
                    x = self.x[n:] + self.x[:i]
                    y_1, y_2 = zip(*[self.__calculate_power__(v) for v in self.y[n:] + self.y[:i]])
                else:
                    x = self.x[n:self.timespan + n]
                    y_1, y_2 = zip(*[self.__calculate_power__(v) for v in self.y[n:self.timespan + n]])
                y_true.append(y_1)
                y_ref.append(y_2)
                x_all.append(x)
            return x_all, y_true, y_ref

    def __propagate__(self):
        for n in self.graph.nodes:
            pass

    def __calculate_power__(self, ref_power_value):
        # algorithm to calculate the new demand for each node
        original_power_usage = 0
        for n in self.graph.nodes:
            self.graph.nodes[n][const.POWER_USAGE_ATTR] = np.random.normal(self.mu, self.sigma) * ref_power_value
            original_power_usage += np.random.normal(self.mu, self.sigma) * ref_power_value
        return util.sum_demand(self.graph) / self.reduce_factor, original_power_usage / self.reduce_factor

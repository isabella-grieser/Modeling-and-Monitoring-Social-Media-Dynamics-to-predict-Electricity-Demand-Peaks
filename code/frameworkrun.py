import json

import matplotlib
from data.social_media.process_sqlite import *
from scipy import signal
from datetime import datetime
import datetime as dt
from framework import EstimationFramework

matplotlib.use("TkAgg")


def scenario1():
    data = get_typhoon_data()

    values = data.groupby(pd.Grouper(key="date", freq="15min"))["tweet"].count()

    start = values.index[0]
    spread_start = datetime(2023, 11, 9, 5, 30, 0, tzinfo=dt.timezone.utc)
    y = signal.savgol_filter(values.values, 53, 3)

    with open("./config/demand-response.json", "r") as f:
        config = json.load(f)

    for s in config["seeds"]:
        config["seed"] = s
        framework = EstimationFramework(config)

        x_start, x_all, y_true, y_ref = framework.estimate_power_outage(start, spread_start=spread_start, data=y)
        val = True


def scenario2():
    start = None
    with open("./config/conspiracy.json", "r") as f:
        config = json.load(f)
    framework = EstimationFramework(config)
    framework.estimate_power_outage(start)


def scenario3():
    start = None
    with open("./config/conspiracy.json", "r") as f:
        config = json.load(f)
    framework = EstimationFramework(config)
    framework.estimate_power_outage(start)


if __name__ == "__main__":
    scenario1()
    # scenario2()
    # scenario3()

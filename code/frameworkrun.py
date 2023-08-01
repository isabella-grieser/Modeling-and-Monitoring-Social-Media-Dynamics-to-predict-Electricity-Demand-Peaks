import matplotlib
from data.social_media.process_sqlite import *
from scipy import signal
matplotlib.use("TkAgg")
from framework import EstimationFramework


if __name__ == "__main__":

    data = get_typhoon_data()

    values = data.groupby(pd.Grouper(key="date", freq="45min"))["tweet"].count()

    values = values[93:]
    start = values.index[93]
    y = signal.savgol_filter(values.values, 53, 3)

    framework = EstimationFramework()


    framework.estimate_power_outage(values, start)
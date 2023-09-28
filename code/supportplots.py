import matplotlib.pyplot as plt
from demandlib import bdew


def utility_companies_plot():
    labels = ['RWE', 'LEAG', 'EnBW', 'E.ON', 'Vattenfall', 'Other']
    sizes = [25.3, 14.9, 9.9, 9.6, 5.6, 34.7]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    fig.show()

def ref_data_plot():
    load_profile = bdew.elec_slp.ElecSlp(2013)
    df = load_profile.get_profile({'h0': 2013})
    x = df.index.to_list()
    y = df.h0.apply(lambda h: h * 100).to_list()
    plt.plot(x, y)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=True)
    plt.show()

if __name__ == "__main__":
    #utility_companies_plot()
    ref_data_plot()
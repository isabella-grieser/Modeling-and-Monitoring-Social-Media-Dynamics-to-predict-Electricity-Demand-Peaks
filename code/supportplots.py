import matplotlib.pyplot as plt


def utility_companies_plot():
    labels = ['RWE', 'LEAG', 'EnBW', 'E.ON', 'Vattenfall', 'Other']
    sizes = [25.3, 14.9, 9.9, 9.6, 5.6, 34.7]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    fig.show()


if __name__ == "__main__":
    utility_companies_plot()

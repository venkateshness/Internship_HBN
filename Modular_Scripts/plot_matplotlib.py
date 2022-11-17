import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def plot_matplotlib(x, y, title, xlabel, ylabel, color):
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

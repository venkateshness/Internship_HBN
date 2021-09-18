import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()


def plot(x, y, xlabel=None, ylabel=None, title=None, color=None,  top=None, bottom=None, axis=None, fill_between=False, marker=None, ls=None):
    plt.figure()
    plt.plot(x, y, linewidth=2, marker=marker, ls=ls, color=color)
    if fill_between:
        plt.fill_between(x, bottom, top, color='b', alpha=.1)
        # plt.axis(axis)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

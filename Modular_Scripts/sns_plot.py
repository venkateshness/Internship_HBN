import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()


def plot(
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    color=None,
    top=None,
    bottom=None,
    axis=None,
    fill_between=False,
    marker=None,
    ls=None,
):
    """ A single script to serve all seaborn-related plots

    Args:
        x ([x]): [x axis data]
        y ([type]): [y axis data]
        xlabel ([type], optional): [x axis label]. Defaults to None.
        ylabel ([type], optional): [y axis label]. Defaults to None.
        title ([type], optional): [Title of the plot]. Defaults to None.
        color ([type], optional): [Color if necessary]. Defaults to None.
        top ([type], optional): [upper bound for CI]. Defaults to None.
        bottom ([type], optional): [lower bound for CI]. Defaults to None.
        axis ([type], optional): [description]. Defaults to None.
        fill_between (bool, optional): [description]. Defaults to False.
        marker ([type], optional): [description]. Defaults to None.
        ls ([type], optional): [description]. Defaults to None.
    """ """"""
    plt.figure()
    plt.plot(x, y, linewidth=2, marker=marker, ls=ls, color=color)
    if fill_between:
        plt.fill_between(x, bottom, top, color="b", alpha=0.1)
        # plt.axis(axis)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

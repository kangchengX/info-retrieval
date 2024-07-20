import matplotlib.pyplot as plt
import numpy as np

def visualize_frequency_zipfian(
        fre_norm : np.ndarray, 
        zipfian : np.ndarray, 
        filename : str | None = None, 
        show : bool | None = True
):
    '''Plot the normalized frequency and zipfian values both in normal (linear) and log-log scales

    Args:
        fre_norm: a vector of normalized frequencies with descending order
        zipfian: a vector of zipfian values with descending order
        filename: None or the filename to save the plot. None indicates not saving the plot
        show: True indicates show the plot

    Returns:
        fig: the Figure object of this plot

    Files Created:
        a file with filename if filename is not None, showing the distrubution and zipfian comparsion both in log-log and normal scale
    '''

    x = np.arange(1,len(fre_norm)+1)

    fig, axes = plt.subplots(1,2,figsize = (14, 6))

    # normal scale
    axes[0].plot(x,fre_norm,'b-',markersize=5,markerfacecolor='none', label = 'empirical')
    axes[0].plot(x,zipfian,'r--',markersize=5,markerfacecolor='none', label = "Zipf's law")
    axes[0].set_title('distribution comparison - normal')

    # log-log scale
    axes[1].loglog(x,fre_norm,'b-',markersize=5,markerfacecolor='none', label = 'empirical')
    axes[1].loglog(x,zipfian,'r--',markersize=5,markerfacecolor='none', label = "Zipf's law")
    axes[1].set_title('distribution comparison - log-log')

    for ax in axes:
        ax.set_xlabel(r'term frequency rank $k$')
        ax.set_ylabel(r'normalized frequency $f$')
        ax.legend()

    if filename is not None:
        fig.savefig(filename)

    if show:
        plt.show()
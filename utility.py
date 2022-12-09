from matplotlib import pyplot as plt
import matplotlib.colors as colors

def plotStyle():
    # Plotting parameters
    # PRL Font preference: computer modern roman (cmr), medium weight (m), normal shape
    cm_in_inch = 2.54
    # column size is 8.6 cm
    col_size = 8.6 / cm_in_inch
    default_width = 1.0*col_size
    aspect_ratio = 5/7
    default_height = aspect_ratio*default_width
    plot_params = {
        'backend': 'pdf',
        'savefig.format': 'pdf',
        'text.usetex': True,
        'font.size': 7,

        'figure.figsize': [default_width, default_height],
        'figure.facecolor': 'white',

        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',

        'axes.titlesize': 8.0,
        'axes.titlepad' : 5,
        'axes.labelsize': 8,
        'legend.fontsize': 6.5,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'axes.linewidth': 0.75,

        'xtick.top': False,
        'xtick.bottom': True,
        'xtick.direction': 'out',
        'xtick.minor.size': 2,
        'xtick.minor.width': 0.5,
        'xtick.major.pad': 2,
        'xtick.major.size': 4,
        'xtick.major.width': 1,

        'ytick.left': True,
        'ytick.right': False,
        'ytick.direction': 'out',
        'ytick.minor.size': 2,
        'ytick.minor.width': 0.5,
        'ytick.major.pad': 2,
        'ytick.major.size': 4,
        'ytick.major.width': 1,

        'lines.linewidth': 1
    }
    plt.rcParams.update(plot_params)
from matplotlib import pyplot as plt
import matplotlib.colors as colors

def plotStyle(style='pres_params'):
    # Plotting parameters
    # PRL Font preference: computer modern roman (cmr), medium weight (m), normal shape
    cm_in_inch = 2.54
    # column size is 8.6 cm
    col_size = 8.6 / cm_in_inch
    default_width = 1.0*col_size
    aspect_ratio = 5/7
    default_height = aspect_ratio*default_width
    textwidth = 6.47699
    
    plot_params_dict = {}
    plot_params_dict['plot_params'] = {
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
    plot_params_dict['pres_params'] = {'axes.edgecolor': 'black',
                      'axes.facecolor':'white',
                      'axes.grid': False,
                      'axes.linewidth': 0.5,
                      'backend': 'ps',
                      'savefig.format': 'pdf',
                      'axes.titlesize': 24,
                      'axes.labelsize': 20,
                      'legend.fontsize': 20,
                      'xtick.labelsize': 18,
                      'ytick.labelsize': 18,
                      'text.usetex': True,
                      'figure.figsize': [7, 5],
                      'font.family': 'sans-serif',
                      #'mathtext.fontset': 'cm',
                      'xtick.bottom':True,
                      'xtick.top': False,
                      'xtick.direction': 'out',
                      'xtick.major.pad': 3,
                      'xtick.major.size': 3,
                      'xtick.minor.bottom': False,
                      'xtick.major.width': 0.2,

                      'ytick.left':True,
                      'ytick.right':False,
                      'ytick.direction':'out',
                      'ytick.major.pad': 3,
                      'ytick.major.size': 3,
                      'ytick.major.width': 0.2,
                      'ytick.minor.right':False,
                      'lines.linewidth':2}

    plot_params_dict['params'] = {'axes.edgecolor': 'black',
                      'axes.facecolor':'white',
                      'axes.grid': False,
                      'axes.linewidth': 0.5,
                      'backend': 'ps',
                      'savefig.format': 'ps',
                      'axes.titlesize': 11,
                      'axes.labelsize': 9,
                      'legend.fontsize': 9,
                      'xtick.labelsize': 8,
                      'ytick.labelsize': 8,
                      'text.usetex': True,
                      'figure.figsize': [7, 5],
                      'font.family': 'sans-serif',
                      #'mathtext.fontset': 'cm',
                      'xtick.bottom':True,
                      'xtick.top': False,
                      'xtick.direction': 'out',
                      'xtick.major.pad': 3,
                      'xtick.major.size': 3,
                      'xtick.minor.bottom': False,
                      'xtick.major.width': 0.2,

                      'ytick.left':True,
                      'ytick.right':False,
                      'ytick.direction':'out',
                      'ytick.major.pad': 3,
                      'ytick.major.size': 3,
                      'ytick.major.width': 0.2,
                      'ytick.minor.right':False,
                      'lines.linewidth':2}
    plt.rcParams.update(plot_params_dict[style])
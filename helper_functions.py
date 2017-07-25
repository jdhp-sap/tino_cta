import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('seaborn-talk')
# plt.style.use('t_slides')

import signal
class SignalHandler():
    ''' handles ctrl+c signals; set up via
        signal_handler = SignalHandler()
        signal.signal(signal.SIGINT, signal_handler)
        # or for two step interupt:
        signal.signal(signal.SIGINT, signal_handler.stop_drawing)
    '''
    def __init__(self):
        self.stop = False
        self.draw = True

    def __call__(self, signal, frame):
        if self.stop:
            print('you pressed Ctrl+C again -- exiting NOW')
            exit(-1)
        print('you pressed Ctrl+C!')
        print('exiting after current event')
        self.stop = True

    def stop_drawing(self, signal, frame):
        if self.stop:
            print('you pressed Ctrl+C again -- exiting NOW')
            exit(-1)

        if self.draw:
            print('you pressed Ctrl+C!')
            print('turn off drawing')
            self.draw = False
        else:
            print('you pressed Ctrl+C!')
            print('exiting after current event')
            self.stop = True


def apply_mc_calibration_ASTRI(adcs, gains, peds, adc_tresh=3500):
    """
    apply basic calibration for ASTRI telescopes with two gains
    """

    calibrated = [(adc0-ped0)*gain0 if adc0 < adc_tresh
                  else (adc1-ped1)*gain1
                  for adc0, adc1, gain0, gain1, ped0, ped1
                  in zip(*adcs, *gains, *peds)]

    return np.array(calibrated)


def apply_mc_calibration(adcs, gains, peds):
    """
    apply basic calibration
    """

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains


def convert_astropy_array(arr, unit=None):
    if unit is None:
        unit = arr[0].unit
    return np.array([a.to(unit).value for a in arr])*unit


def make_argparser():
    from os.path import expandvars
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--max_events', type=int, default=None,
                        help="maximum number of events considered per file")
    parser.add_argument('-c', '--min_charge', type=int, default=0,
                        help="minimum charge per telescope after cleaning")
    parser.add_argument('-i', '--indir',   type=str,
                        default=expandvars("$HOME/Data/cta/ASTRI9/"))
                        # default="/media/tmichael/Transcend/Data/cta/ASTRI9/")
    parser.add_argument('-f', '--infile_list',   type=str, default="", nargs='*',
                        help="give a specific list of files to run on")
    parser.add_argument('--plots_dir', type=str, default="plots",
                        help="path to store plots")
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave",
                        help="if set, use tail cleaning, otherwise wavelets")
    parser.add_argument('--dilate', default=False, action='store_true',
                        help="use dilate function for tailcut cleaning")
    parser.add_argument('--no_reject_edge', dest='skip_edge_events', default=True,
                        action='store_false', help="do not reject edge events")
    parser.add_argument('-w', '--write', action='store_true',
                        help="write summary-level output -- e.g. plots, tables")
    parser.add_argument('--store', action='store_true',
                        help="write event data / trained classifier")
    parser.add_argument('-p', '--plot',  action='store_true',
                        help="do some plotting")
    parser.add_argument('-v', '--verbose',  action='store_true',
                        help="do things more explicit -- plotting, logging etc.")
    parser.add_argument('-d', '--dry', dest='last', action='store_const',
                        const=1, default=None,
                        help="only consider first file per type")
    parser.add_argument('--raw', type=str, default=None,
                        help="raw option string for wavelet filtering")
    return parser


try:
    from matplotlib2tikz import save as tikzsave

    def tikz_save(arg, **kwargs):
        tikzsave(arg, **kwargs,
                 figureheight='\\figureheight',
                 figurewidth='\\figurewidth')
except:
    print("matplotlib2tikz is not installed")
    print("install with: \n$ pip install matplotlib2tikz")

    def tikz_save(arg, **kwargs):
        print("matplotlib2tikz is not installed")
        print("no .tex is saved")


def save_fig(outname, endings=["tex", "pdf", "png"], **kwargs):
    for end in endings:
        if end == "tex":
            tikz_save("{}.{}".format(outname, end), **kwargs)
        else:
            plt.savefig("{}.{}".format(outname, end))


def plot_hex_and_violin(abscissa, ordinate, bin_edges, extent=None,
                        xlabel="", ylabel="", zlabel="", do_hex=True, do_violin=True,
                        cm=plt.cm.inferno, **kwargs):

    """
    takes two arrays of coordinates and creates a 2D hexbin plot and a violin plot (or
    just one of them)

    Parameters
    ----------
    abscissa, ordinate : arrays
        the coordinates of the data to plot
    bin_edges : array
        bin edges along the abscissa
    extent : 4-tuple of floats (default: None)
        extension of the abscissa, ordinate; given as is to plt.hexbin
    xlabel, ylabel : strings (defaults: "")
        labels for the two axes of either plot
    zlabel : string (default: "")
        label for the colorbar of the hexbin plot
    do_hex, do_violin : bools (defaults: True)
        whether or not to do the respective plots
    cm : colour map (default: plt.cm.inferno)
        colour map to be used for the hexbin plot
    kwargs : args dictionary
        more arguments to be passed to plt.hexbin
    """

    plt.figure()

    # make a normal 2D hexplot from the given data
    if do_hex:

        # if we do both plot types, open a subplot
        if do_violin:
            plt.subplot(211)

        plt.hexbin(abscissa,
                   ordinate,
                   gridsize=40,
                   extent=extent,
                   cmap=cm,
                   **kwargs)
        cb = plt.colorbar()
        cb.set_label(zlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if extent:
            plt.xlim(extent[:2])
            plt.ylim(extent[2:])

    # prepare and draw the data for the violin plot
    if do_violin:

        # if we do both plot types, open a subplot
        if do_hex:
            plt.subplot(212)

        # to plot the violins, sort the ordinate values into a dictionary
        # the keys are the central values of the bins given by `bin_edges`
        val_vs_dep = {}
        bin_centres = (bin_edges[1:]+bin_edges[:-1])/2.
        for dep, val in zip(abscissa, ordinate):
            # get the bin number this event belongs into
            # outliers are put into the first and last bin accordingly
            ibin = np.clip(np.digitize(dep, bin_edges)-1,
                           0, len(bin_centres)-1)

            # the central value of the bin is the key for the dictionary
            if bin_centres[ibin] not in val_vs_dep:
                val_vs_dep[bin_centres[ibin]]  = [val]
            else:
                val_vs_dep[bin_centres[ibin]] += [val]

        vals = [a for a in val_vs_dep.values()]
        keys = [a for a in val_vs_dep.keys()]

        # calculate the widths of the violins as 90 % of the corresponding bin width
        widths = []
        for cen, wid in zip(bin_centres, (bin_edges[1:]-bin_edges[:-1])):
            if cen in keys:
                widths.append(wid*.9)

        plt.violinplot(vals, keys,
                       points=60, widths=widths,
                       showextrema=False, showmedians=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if extent:
            # adding a colour bar to the hexbin plot reduces its width by 1/5
            # adjusting the extent of the violin plot to sync up with the hexbin plot
            plt.xlim([extent[0], (5.*extent[1] - extent[0])/4.])
            # for good measure also sync the vertical extent
            plt.ylim(extent[2:])

        plt.grid()


def ipython_shell():
    # doesn't actually work, needs to be put inline, here only as a reminder
    from IPython import embed
    embed()

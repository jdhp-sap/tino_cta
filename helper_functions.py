from os.path import expandvars

import numpy as np

from astropy import units as u

from matplotlib import pyplot as plt

import signal
class SignalHandler():
    ''' handles ctrl+c signals; set up via
        signal_handler = SignalHandler()
        signal.signal(signal.SIGINT, signal_handler)
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


def apply_mc_calibration_ASTRI(adcs, gains, peds, mode=0, adc_tresh=3500):
    """
    apply basic calibration for ASTRI telescopes with two gains
    """
    gains0 = gains[0]
    gains1 = gains[1]

    peds0 = peds[0]
    peds1 = peds[1]

    calibrated = [(adc0-ped0)*gain0 if adc0 < adc_tresh
                    else (adc1-ped1)*gain1
                    for adc0, adc1, gain0, gain1, ped0, ped1
                    in zip(adcs[0], adcs[1], gains0, gains1, peds0, peds1)]

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
        return (np.array([a.to(unit).value for a in arr])*unit).si
    else:
        return np.array([a.to(unit).value for a in arr])*unit


import argparse
def make_argparser():
    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max_events', type=int, default=None,
                        help="maximum number of events considered per file")
    parser.add_argument('-c', '--min_charge', type=int, default=0,
                        help="minimum charge per telescope after cleaning")
    parser.add_argument('-i', '--indir',   type=str,
                        default=expandvars("$HOME/Data/cta/ASTRI9/"))
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave",
                        help="if set, use tail cleaning, otherwise wavelets")
    parser.add_argument('--dilate', default=False, action='store_true',
                        help="use dilate function for tailcut cleaning")
    parser.add_argument('-w', '--write', action='store_true',
                        help="write output -- e.g. plots, classifiers, events")
    parser.add_argument('-p', '--plot',  action='store_true',
                        help="do some plotting")
    parser.add_argument('-d', '--dry', dest='last', action='store_const',
                        const=1, default=None,
                        help="only consider first file per type")
    return parser


try:
    from matplotlib2tikz import save as tikzsave
except:
    print("matplotlib2tikz is not installed")

def tikz_save(arg, **kwargs):
    try:
        tikzsave(arg, figureheight = '\\figureheight',
                      figurewidth  = '\\figurewidth', **kwargs)
    except:
        print("matplotlib2tikz is not installed")


def save_fig(outname, endings=["tex", "pdf", "png"], **kwargs):
    for end in endings:
        if end == "tex":
            tikz_save("{}.{}".format(outname, end), **kwargs)
        else:
            ptl.savefig("{}.{}".format(outname, end))


def make_mock_event_rate(spectra, bin_edges=None, Emin=None, Emax=None,
                         E_unit=u.GeV, NBins=None, logE=True, norm=None):

    rates = [[] for f in spectra]

    if bin_edges is None:
        if logE:
            Emin = np.log10(Emin/E_unit)
            Emax = np.log10(Emax/E_unit)
        bin_edges = np.linspace(Emin, Emax, NBins+1, True)

    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        if logE:
            bin_centre = 10**((l_edge+h_edge)/2.) * E_unit
            bin_width = (10**h_edge-10**l_edge)*E_unit

        else:
            bin_centre = (l_edge+h_edge) * E_unit / 2.
            bin_width = (h_edge-l_edge)*E_unit
        for i, spectrum in enumerate(spectra):
            bin_events = spectrum(bin_centre) * bin_width
            rates[i].append(bin_events)

    for i, rate in enumerate(rates):
        rate = convert_astropy_array(rate)
        if norm:
            rate *= norm[i]/np.sum(rate)
        rates[i] = rate

    return (*rates), bin_edges

# ================================== #
# Compute Eq. (17) of Li & Ma (1983) #
# ================================== #
def sigma_lima(Non, Noff, alpha=0.2):
    """
    Compute Eq. (17) of Li & Ma (1983).

    Parameters:
     Non   - Number of on counts
     Noff  - Number of off counts
    Keywords:
     alpha - Ratio of on-to-off exposure
    """

    alpha1 = alpha + 1.0
    sum    = Non + Noff
    arg1   = Non / sum
    arg2   = Noff / sum
    term1  = Non  * np.log((alpha1/alpha)*arg1)
    term2  = Noff * np.log(alpha1*arg2)
    sigma  = np.sqrt(2.0 * (term1 + term2))

    return sigma




def plot_hex_and_violin(abscissa, ordinate, bin_edges, extent=None, vmin=None, vmax=None,
                         xlabel="", ylabel="", zlabel="", do_hex=True, do_violin=True,
                         cm=plt.cm.hot, **kwargs):

    """
    takes two arrays of coordinates and creates a 2D hexbin plot and a violin plot (or
    just one of them)

    Parameters:
    -----------
    abscissa, ordinate : arrays
        the coordinates of the data to plot
    bin_edges : array
        bin edges along the abscissa
    extent : 4-tuple of floats (default: None)
        extension of the abscissa, ordinate; given as is to plt.hexbin
    vmin, vmax : floats (defaults: None)
        lower and upper caps of the bin values to be plotted in plt.hexbin
    xlabel, ylabel : strings (defaults: "")
        labels for the two axes of either plot
    zlabel : string (default: "")
        label for the colorbar of the hexbin plot
    do_hex, do_violin : bools (defaults: True)
        whether or not to do the respective plots
    cm : colour map (default: plt.cm.hot)
        colour map to be used for the hexbin plot
    kwargs : args dictionary
        more arguments to be passed to plt.hexbin
    """


    plt.figure()

    ''' make a normal 2D hexplot from the given data '''
    if do_hex:

        ''' if we do both plot types, open a subplot '''
        if do_violin:
            plt.subplot(211)

        plt.hexbin(abscissa,
                   ordinate,
                   vmin=vmin,
                   vmax=vmax,
                   gridsize=40,
                   extent=extent,
                   cmap=cm,
                   **kwargs)
        cb = plt.colorbar()
        cb.set_label(zlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    ''' prepare and draw the data for the violin plot '''
    if do_violin:

        ''' if we do both plot types, open a subplot '''
        if do_hex:
            plt.subplot(212)

        '''
        to plot the violins, sort the ordinate values into a dictionary
        the keys are the central values of the bins given by @bin_edges '''
        val_vs_dep = {}
        bin_centres = (bin_edges[1:]+bin_edges[:-1])/2.
        for dep, val in zip(abscissa, ordinate):
            ''' get the bin number this event belongs into '''
            ibin = np.clip(
                        np.digitize(dep, bin_edges)-1,
                        0, len(bin_centres)-1)

            ''' the central value of the bin is the key for the dictionary '''
            if bin_centres[ibin] not in val_vs_dep:
                val_vs_dep[bin_centres[ibin]]  = [val]
            else:
                val_vs_dep[bin_centres[ibin]] += [val]

        vals = [a for a in val_vs_dep.values()]
        keys = [a for a in val_vs_dep.keys()]

        '''
        calculate the widths of the violins as 90 % of the corresponding bin width '''
        widths=[]
        for cen, wid in zip(bin_centres, (bin_edges[1:]-bin_edges[:-1])):
            if cen in keys:
                widths.append(wid*.9)


        plt.violinplot(vals, keys,
                       points=60, widths=widths,
                       showextrema=True, showmedians=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ''' adding a colour bar to tex hexbin plot reduces its total width to 4/5
        adjusting the extent of the violin plot to sync up with the hexbin plot '''
        if not np.isnan(extent[:2]).any():
            plt.xlim([extent[0], (5.*extent[1] - extent[0])/4.])
        ''' for good measure also sync the vertical extent '''
        if not np.isnan(extent[2:]).any():
            plt.ylim(extent[2:])
        plt.grid()


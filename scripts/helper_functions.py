import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
# plt.style.use('t_slides')

import signal


class SignalHandler():
    ''' handles ctrl+c signals; set up via
        `signal_handler = SignalHandler()
        `signal.signal(signal.SIGINT, signal_handler)`
        # or for two step interupt:
        `signal.signal(signal.SIGINT, signal_handler.stop_drawing)`
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


def convert_astropy_array(arr, unit=None):
    if unit is None:
        unit = arr[0].unit
    return np.array([a.to(unit).value for a in arr]) * unit


def make_argparser():
    from os.path import expandvars
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--max_events', type=int, default=None,
                        help="maximum number of events considered per file")
    parser.add_argument('-c', '--min_charge', type=int, default=0,
                        help="minimum charge per telescope after cleaning")
    parser.add_argument('-i', '--indir', type=str,
                        default=expandvars("$HOME/Data/cta/Prod3b/Paranal"))
    parser.add_argument('-f', '--infile_list', type=str, default="", nargs='*',
                        help="give a specific list of files to run on")
    parser.add_argument('--cam_ids', type=str,
                        default=["LSTCam", "NectarCam", "DigiCam"], nargs='*',
                        help="give a specific list of files to run on")
    parser.add_argument('--plots_dir', type=str, default="plots/",
                        help="path to store plots")
    parser.add_argument('--dilate', default=False, action='store_true',
                        help="use dilate function for tailcut cleaning")
    parser.add_argument('--no_reject_edge', dest='skip_edge_events', default=True,
                        action='store_false', help="do not reject edge events")
    parser.add_argument('-w', '--write', action='store_true',
                        help="write summary-level output -- e.g. plots, tables")
    parser.add_argument('--store', action='store_true',
                        help="write event data / trained classifier")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="do some plotting")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="do things more explicit -- plotting, logging etc.")
    parser.add_argument('-d', '--dry', dest='last', action='store_const',
                        const=1, default=None,
                        help="only consider first file per type")
    parser.add_argument('--raw', type=str, default=None,
                        help="raw option string for wavelet filtering")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--wave', dest="mode", action='store_const',
                            const="wave", default="wave",
                            help="if set, use wavelet cleaning -- default")
    mode_group.add_argument('--tail', dest="mode", action='store_const',
                            const="tail",
                            help="if set, use tail cleaning, otherwise wavelets")

    return parser


try:
    from matplotlib2tikz import save as tikzsave

    def tikz_save(arg, **kwargs):
        tikzsave(arg, **kwargs,
                 figureheight='\\figureheight',
                 figurewidth='\\figurewidth')
except ImportError:
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
                        cm=plt.cm.inferno, axis=None, v_padding=.015, **kwargs):

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

    if axis:
        if do_hex and do_violin:
            from matplotlib.axes import Axes
            from matplotlib.transforms import Bbox
            axis_bbox = axis.get_position()
            axis.axis("off")
        else:
            plt.sca(axis)

    # make a normal 2D hexplot from the given data
    if do_hex:

        # if we do both plot types,
        if do_violin:
            if axis:
                ax_hex_pos = axis_bbox.get_points().copy()  # [[x0, y0], [x1, y1]]
                ax_hex_pos[0, 1] += np.diff(ax_hex_pos, axis=0)[0, 1]*(.5+v_padding)
                ax_hex = Axes(plt.gcf(), Bbox.from_extents(ax_hex_pos))
                plt.gcf().add_axes(ax_hex)
                plt.sca(ax_hex)
                ax_hex.set_xticklabels([])
            else:
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
            if axis:
                ax_vio_pos = axis_bbox.get_points().copy()  # [[x0, y0], [x1, y1]]
                ax_vio_pos[1, 1] -= np.diff(ax_vio_pos, axis=0)[0, 1]*(.5+v_padding)
                ax_vio = Axes(plt.gcf(), Bbox.from_extents(ax_vio_pos))
                plt.gcf().add_axes(ax_vio)
                plt.sca(ax_vio)
            else:
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
                val_vs_dep[bin_centres[ibin]] = [val]
            else:
                val_vs_dep[bin_centres[ibin]] += [val]

        keys = [k[0] for k in sorted(val_vs_dep.items())]
        vals = [k[1] for k in sorted(val_vs_dep.items())]

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
            plt.xlim([extent[0],
                      (5.*extent[1] - extent[0])/4. if do_hex else extent[1]])
            # for good measure also sync the vertical extent
            plt.ylim(extent[2:])

        plt.grid()


def prod3b_tel_ids(cam_id, site="south"):
    if cam_id in [None, ""]:
        return None

    if site.lower() in ["south", "paranal", "chile"]:
        if cam_id == "LSTCam":
            tel_ids = np.arange(12)
        elif cam_id == "FlashCam":
            tel_ids = np.arange(12, 53)
        elif cam_id == "NectarCam":
            tel_ids = np.arange(53, 94)
        elif cam_id == "ASTRICam":
            tel_ids = np.arange(95, 252)
        elif cam_id == "CHEC":
            tel_ids = np.arange(252, 410)
        elif cam_id == "DigiCam":
            tel_ids = np.arange(410, 567)

        elif cam_id == "L+F+A":
            tel_ids = np.array([4, 5, 6, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24,
                                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48,
                                49, 50, 51, 52, 99, 100, 101, 102, 110, 111, 116,
                                117, 122, 123, 124, 125, 126, 127, 132, 133, 134,
                                135, 142, 143, 144, 145, 158, 159, 164, 165, 166,
                                167, 169, 170, 184, 185, 186, 187, 188, 189, 190,
                                191, 192, 193, 194, 195, 208, 209, 210, 211, 212,
                                213, 220, 221, 222, 223, 224, 225, 226, 227, 228,
                                229, 234, 235, 236, 237, 238, 239, 240, 241, 242,
                                243, 244, 245])
        elif cam_id == "F+A":
            tel_ids = np.array([12, 13, 14, 15, 16, 19, 20, 23, 24,
                                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48,
                                49, 50, 51, 52, 99, 100, 101, 102, 110, 111, 116,
                                117, 122, 123, 124, 125, 126, 127, 132, 133, 134,
                                135, 142, 143, 144, 145, 158, 159, 164, 165, 166,
                                167, 169, 170, 184, 185, 186, 187, 188, 189, 190,
                                191, 192, 193, 194, 195, 208, 209, 210, 211, 212,
                                213, 220, 221, 222, 223, 224, 225, 226, 227, 228,
                                229, 234, 235, 236, 237, 238, 239, 240, 241, 242,
                                243, 244, 245])

        elif cam_id == "L+F+D":
            tel_ids = np.array([4, 5, 6, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26,
                                27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50, 51, 52,
                                415, 416, 417, 418, 426, 427, 432, 433, 438, 439, 440,
                                441, 442, 443, 448, 449, 450, 451, 458, 459, 460, 461,
                                474, 475, 480, 481, 482, 483, 485, 486, 500, 501, 502,
                                503, 504, 505, 506, 507, 508, 509, 510, 511, 524, 525,
                                526, 527, 528, 529, 536, 537, 538, 539, 540, 541, 542,
                                543, 544, 545, 550, 551, 552, 553, 554, 555, 556, 557,
                                558, 559, 560, 561])

        elif cam_id == "L+N+D":
            tel_ids = np.array([4, 5, 6, 11, 53, 54, 55, 56, 57, 60, 61, 64, 65, 66, 67,
                                68, 69, 70, 71, 72, 73, 74, 75, 88, 89, 90, 91, 92, 93,
                                415, 416, 417, 418, 426, 427, 432, 433, 438, 439, 440,
                                441, 442, 443, 448, 449, 450, 451, 458, 459, 460, 461,
                                474, 475, 480, 481, 482, 483, 485, 486, 500, 501, 502,
                                503, 504, 505, 506, 507, 508, 509, 510, 511, 524, 525,
                                526, 527, 528, 529, 536, 537, 538, 539, 540, 541, 542,
                                543, 544, 545, 550, 551, 552, 553, 554, 555, 556, 557,
                                558, 559, 560, 561])

        else:
            raise ValueError("cam_id {} not supported".format(cam_id))
    elif site.lower in ["north", "la palma", "lapalma", "spain", "canaries"]:
        raise ValueError("north site not implemented yet")
    else:
        raise ValueError("site '{}' not known -- try again".format(site))

    return tel_ids


def ipython_shell():
    # doesn't actually work, needs to be put inline, here only as a reminder
    from IPython import embed
    embed()


import inspect
def lineno():
    """Returns the current filename and line number"""
    frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
    return ", ".join([frameinfo.filename,
                      str(frameinfo.lineno)])


def sliding_window(my_list, window_size, step_size=None, start=0):
    step_size = step_size or window_size
    while start+window_size < len(my_list):
        yield my_list[start:start+window_size]
        start += step_size
    else:
        yield my_list[start:]

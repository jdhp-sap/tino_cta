#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import exit
import numpy as np

from glob import glob

import matplotlib.pyplot as plt

from astropy import units as u

try:
    import tables as tb
except:
    # if pytables are not installed, build a class that mimicks the interface but
    # immediately throws an exception that we can catch. this way, we get still thrown
    # off by e.g. naming errors but not by a missing "reco_event"
    class NoPyTError(Exception):
        pass

    class RecoEvent(dict):
        def __setitem__(self, foo, bar):
            raise NoPyTError
    print("no pytables installed")

from ctapipe.io.hessio import hessio_event_source

from ctapipe.utils import linalg
from ctapipe.utils.CutFlow import CutFlow

from ctapipe.image.hillas import HillasParameterizationError, \
                                 hillas_parameters_4 as hillas_parameters

from ctapipe.reco.HillasReconstructor import \
    HillasReconstructor, TooFewTelescopes
from ctapipe.reco.shower_max import ShowerMaxEstimator

from ctapipe.coordinates.coordinate_transformations import az_to_phi, alt_to_theta

from modules.ImageCleaning import ImageCleaner

from modules.prepare_event import EventPreparer

from helper_functions import *


def main():

    # your favourite units here
    energy_unit = u.TeV
    angle_unit = u.deg
    dist_unit = u.m

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/reconstructed_events")
    parser.add_argument('-o', '--out_file', type=str, default="rec_events")
    parser.add_argument('--plot_c',  action='store_true',
                        help="plot camera-wise displays")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--proton',  action='store_true',
                       help="do protons instead of gammas")
    group.add_argument('--electron',  action='store_true',
                       help="do electrons instead of gammas")

    args = parser.parse_args()

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
    elif args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
    elif args.electron:
        filenamelist = glob("{}/electron/*gz".format(args.indir))
    else:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))
    filenamelist.sort()

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    tel_phi = {}
    tel_theta = {}

    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # takes care of image cleaning
    cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           skip_edge_events=args.skip_edge_events, island_cleaning=True)

    # the class that does the shower reconstruction
    shower_reco = HillasReconstructor()

    shower_max_estimator = ShowerMaxEstimator("paranal")

    preper = EventPreparer(cleaner=cleaner,
                           hillas_parameters=hillas_parameters, shower_reco=shower_reco,
                           event_cutflow=Eventcutflow, image_cutflow=Imagecutflow,
                           # event/image cuts:
                           allowed_cam_ids=[],  # means: all
                           min_ntel=3,
                           min_charge=args.min_charge,
                           min_pixel=3)

    # a signal handler to abort the event loop but still do the post-processing
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # this class defines the reconstruction parameters to keep track of
        class RecoEvent(tb.IsDescription):
            NTels_trigg = tb.Int16Col(dflt=1, pos=0)
            NTels_clean = tb.Int16Col(dflt=1, pos=1)
            EnMC = tb.Float32Col(dflt=1, pos=2)
            xi = tb.Float32Col(dflt=1, pos=3)
            DeltaR = tb.Float32Col(dflt=1, pos=4)
            ErrEstPos = tb.Float32Col(dflt=1, pos=5)
            ErrEstDir = tb.Float32Col(dflt=1, pos=6)
            h_max = tb.Float32Col(dflt=1, pos=7)

        channel = "gamma" if "gamma" in " ".join(filenamelist) else "proton"
        if args.out_file:
            out_filename = args.out_file
        else:
            out_filename = "classified_events_{}_{}.h5".format(channel, args.mode)
        reco_outfile = tb.open_file(
            "{}/{}".format(args.events_dir, out_filename), mode="w",
            # if we don't want to write the event list to disk, need to add more arguments
            **({} if args.store else {"driver": "H5FD_CORE",
                                      "driver_core_backing_store": False}))
        reco_table = reco_outfile.create_table("/", "reco_event", RecoEvent)
        reco_event = reco_table.row
    except:
        reco_event = RecoEvent()
        print("no pytables installed?")

    # ##        #######   #######  ########
    # ##       ##     ## ##     ## ##     ##
    # ##       ##     ## ##     ## ##     ##
    # ##       ##     ## ##     ## ########
    # ##       ##     ## ##     ## ##
    # ##       ##     ## ##     ## ##
    # ########  #######   #######  ##

    cam_id_map = {}

    # define here which telescopes to loop over
    allowed_tels = None
    # allowed_tels = prod3b_tel_ids("L+F+D")
    for i, filename in enumerate(filenamelist[:args.last]):
        print(f"file: {i} filename = {filename}")

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # loop that cleans and parametrises the images and performs the reconstruction
        for (event, hillas_dict, n_tels,
             tot_signal, max_signal, pos_fit, dir_fit, h_max,
             err_est_pos, err_est_dir) in preper.prepare_event(source):

            shower = event.mc

            org_alt = u.Quantity(shower.alt).to(u.deg)
            org_az = u.Quantity(shower.az).to(u.deg)
            if org_az > 180*u.deg:
                org_az -= 360*u.deg

            org_the = alt_to_theta(org_alt)
            org_phi = az_to_phi(org_az)
            if org_phi > 180*u.deg:
                org_phi -= 360*u.deg
            if org_phi < -180*u.deg:
                org_phi += 360*u.deg

            shower_org = linalg.set_phi_theta(org_phi, org_the)
            shower_core = convert_astropy_array([shower.core_x, shower.core_y])

            xi = linalg.angle(dir_fit, shower_org).to(angle_unit)
            diff = linalg.length(pos_fit[:2]-shower_core)

            # print some performance
            print()
            print("xi = {:4.3f}".format(xi))
            print("pos = {:4.3f}".format(diff))
            print("h_max reco: {:4.3f}".format(h_max.to(u.km)))
            print("err_est_dir: {:4.3f}".format(err_est_dir.to(angle_unit)))
            print("err_est_pos: {:4.3f}".format(err_est_pos))

            try:
                # store the reconstruction data in the PyTable
                reco_event["NTels_trigg"] = n_tels["tot"]
                reco_event["NTels_clean"] = len(shower_reco.circles)
                reco_event["EnMC"] = event.mc.energy / energy_unit
                reco_event["xi"] = xi / angle_unit
                reco_event["DeltaR"] = diff / dist_unit
                reco_event["ErrEstPos"] = err_est_pos / dist_unit
                reco_event["ErrEstDir"] = err_est_dir / angle_unit
                reco_event["h_max"] = h_max / dist_unit
                reco_event.append()
                reco_table.flush()

                print()
                print("xi res (68-percentile) = {:4.3f} {}"
                      .format(np.percentile(reco_table.cols.xi, 68), angle_unit))
                print("core res (68-percentile) = {:4.3f} {}"
                      .format(np.percentile(reco_table.cols.DeltaR, 68), dist_unit))
                print("h_max (median) = {:4.3f} {}"
                      .format(np.percentile(reco_table.cols.h_max, 50), dist_unit))

            except NoPyTError:
                pass

            if args.plot_c:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                for c in shower_reco.circles.values():
                    points = [c.pos + t * c.a * u.km for t in np.linspace(0, 15, 3)]
                    ax.plot(*np.array(points).T, linewidth=np.sqrt(c.weight)/10)
                    ax.scatter(*c.pos[:, None].value, s=np.sqrt(c.weight))
                plt.xlabel("x")
                plt.ylabel("y")
                plt.pause(.1)

                # this plots
                # • the MC shower core
                # • the reconstructed shower core
                # • the used telescopes
                # • and the trace of the Hillas plane on the ground
                plt.figure()
                for tel_id, c in shower_reco.circles.items():
                    plt.scatter(c.pos[0], c.pos[1], s=np.sqrt(c.weight))
                    plt.gca().annotate(tel_id, (c.pos[0].value, c.pos[1].value))
                    plt.plot([c.pos[0].value-500*c.norm[1], c.pos[0].value+500*c.norm[1]],
                             [c.pos[1].value+500*c.norm[0], c.pos[1].value-500*c.norm[0]],
                             linewidth=np.sqrt(c.weight)/10)
                plt.scatter(*pos_fit[:2], c="black", marker="*", label="fitted")
                plt.scatter(*shower_core[:2], c="black", marker="P", label="MC")
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.xlim(-1400, 1400)
                plt.ylim(-1400, 1400)
                plt.show()

            if signal_handler.stop: break
        if signal_handler.stop: break

    print("\n" + "="*35 + "\n")
    print("xi res (68-percentile) = {:4.3f} {}"
          .format(np.percentile(reco_table.cols.xi, 68), angle_unit))
    print("core res (68-percentile) = {:4.3f} {}"
          .format(np.percentile(reco_table.cols.DeltaR, 68), dist_unit))
    print("h_max (median) = {:4.3f} {}"
          .format(np.percentile(reco_table.cols.h_max, 50), dist_unit))

    # print the cutflows for telescopes and camera images
    print("\n\n")
    Eventcutflow("min2Tels trig")
    print()
    Imagecutflow(sort_column=1)

    # if we don't want to plot anything, we can exit now
    if not args.plot:
        return

    # ########  ##        #######  ########  ######
    # ##     ## ##       ##     ##    ##    ##    ##
    # ##     ## ##       ##     ##    ##    ##
    # ########  ##       ##     ##    ##     ######
    # ##        ##       ##     ##    ##          ##
    # ##        ##       ##     ##    ##    ##    ##
    # ##        ########  #######     ##     ######

    plt.figure()
    plt.hist(reco_table.cols.h_max, bins=np.linspace(000, 15000, 51, True))
    plt.title(channel)
    plt.xlabel("h_max reco")
    plt.pause(.1)

    figure = plt.figure()
    xi_edges = np.linspace(0, 5, 20)
    plt.hist(reco_table.cols.xi, bins=xi_edges, log=True)
    plt.xlabel(r"$\xi$ / deg")
    if args.write:
        save_fig('{}/reco_xi_{}'.format(args.plots_dir, args.mode))
    plt.pause(.1)

    plt.figure()
    plt.hist(reco_table.cols.ErrEstDir[:],
             bins=np.linspace(0, 20, 50))
    plt.title(channel)
    plt.xlabel("beta")
    plt.pause(.1)

    plt.figure()
    plt.hist(np.log10(reco_table.cols.xi[:] / reco_table.cols.ErrEstDir[:]), bins=50)
    plt.title(channel)
    plt.xlabel("log_10(xi / beta)")
    plt.pause(.1)

    # convert the xi-list into a dict with the number of used telescopes as keys
    xi_vs_tel = {}
    for xi, ntel in zip(reco_table.cols.xi, reco_table.cols.NTels_clean):
        if ntel not in xi_vs_tel:
            xi_vs_tel[ntel] = [xi]
        else:
            xi_vs_tel[ntel].append(xi)

    print(args.mode)
    for ntel, xis in sorted(xi_vs_tel.items()):
        print("NTel: {} -- median xi: {}".format(ntel, np.median(xis)))
        # print("histogram:", np.histogram(xis, bins=xi_edges))

    # create a list of energy bin-edges and -centres for violin plots
    Energy_edges = np.linspace(2, 8, 13)
    Energy_centres = (Energy_edges[1:]+Energy_edges[:-1])/2.

    # convert the xi-list in to an energy-binned dict with the bin centre as keys
    xi_vs_energy = {}
    for en, xi in zip(reco_table.cols.EnMC, reco_table.cols.xi):

        # get the bin number this event belongs into
        sbin = np.digitize(np.log10(en), Energy_edges)-1

        # the central value of the bin is the key for the dictionary
        if Energy_centres[sbin] not in xi_vs_energy:
            xi_vs_energy[Energy_centres[sbin]]  = [xi]
        else:
            xi_vs_energy[Energy_centres[sbin]] += [xi]

    # plotting the angular error as violin plots with binning in
    # number of telescopes and shower energy
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in xi_vs_tel.values()],
                   [a for a in xi_vs_tel.keys()],
                   points=60, widths=.75, showextrema=False, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\xi$ / deg)")
    plt.ylim(-3, 2)
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in xi_vs_energy.values()],
                   [a for a in xi_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=False, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\xi$ / deg)")
    plt.ylim(-3, 2)
    plt.grid()

    plt.tight_layout()
    if args.write:
        save_fig('{}/reco_xi_vs_E_NTel_{}'.format(args.plots_dir, args.mode))

    plt.pause(.1)

    # convert the diffs-list into a dict with the number of used telescopes as keys
    diff_vs_tel = {}
    for diff, ntel in zip(reco_table.cols.DeltaR, reco_table.cols.NTels_clean):
        if ntel not in diff_vs_tel:
            diff_vs_tel[ntel] = [diff]
        else:
            diff_vs_tel[ntel].append(diff)

    # convert the diffs-list in to an energy-binned dict with the bin centre as keys
    diff_vs_energy = {}
    for en, diff in zip(reco_table.cols.EnMC, reco_table.cols.DeltaR):

        # get the bin number this event belongs into
        sbin = np.digitize(np.log10(en), Energy_edges)-1

        # the central value of the bin is the key for the dictionary
        if Energy_centres[sbin] not in diff_vs_energy:
            diff_vs_energy[Energy_centres[sbin]]  = [diff]
        else:
            diff_vs_energy[Energy_centres[sbin]] += [diff]

    # plotting the core position error as violin plots with binning in
    # number of telescopes an shower energy
    plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in diff_vs_tel.values()],
                   [a for a in diff_vs_tel.keys()],
                   points=60, widths=.75, showextrema=False, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in diff_vs_energy.values()],
                   [a for a in diff_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=False, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()

    plt.tight_layout()
    if args.write:
        save_fig('{}/reco_dist_vs_E_NTel_{}'.format(args.plots_dir, args.mode))
    plt.show()


if __name__ == '__main__':
    main()

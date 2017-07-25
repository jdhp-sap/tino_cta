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
    print("no pytables installed")

from ctapipe.instrument.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.utils import linalg

from ctapipe.image.hillas import HillasParameterizationError, \
                                 hillas_parameters_4 as hillas_parameters

from ctapipe.reco.HillasReconstructor import \
    HillasReconstructor, TooFewTelescopesException


from modules.ImageCleaning import ImageCleaner, EdgeEventException
from modules.CutFlow import CutFlow

from helper_functions import *

from collections import OrderedDict


def main():

    # your favourite units here
    energy_unit = u.GeV
    angle_unit = u.deg
    dist_unit = u.m

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/reconstructed_events")
    parser.add_argument('-o', '--out_file', type=str, default="rec_events")
    parser.add_argument('--photon',  action='store_true',
                        help="use the mc photo-electrons container "
                        "instead of the PMT signal")
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")

    args = parser.parse_args()

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
    elif args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
    else:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))
    filenamelist.sort()

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    Eventcutflow.set_cuts(OrderedDict([
                            ("noCuts", None),
                            ("min2Tels", lambda x: x < 2),
                            ("min2Images", lambda x: x < 2),
                            ("GreatCircles", None),
                            ("nan pos", lambda x: np.isnan(x.value).any()),
                            ("nan dir", lambda x: np.isnan(x.value).any())])
                         )

    min_charge_string = "min charge >= {}".format(args.min_charge)
    Imagecutflow.set_cut(min_charge_string, lambda x: x < args.min_charge)

    # takes care of image cleaning
    Cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           skip_edge_events=False, island_cleaning=True)

    # the class that does the shower reconstruction
    fit = HillasReconstructor()

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
        print("no pytables installed?")

    # define here which telescopes to loop over
    allowed_tels = None
    allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    # allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = range(34, 40)  # use the array of FlashCams instead
    # allowed_tels = np.arange(10).tolist() + np.arange(34, 41).tolist()
    for filename in sorted(filenamelist)[:5]:  # args.last]:

        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            Eventcutflow.count("noCuts")

            if Eventcutflow.cut("min2Tels", len(event.dl0.tels_with_data)):
                continue

            print('Scanning input file... count = {}'.format(event.count))
            print('Event ID: {}'.format(event.dl0.event_id))
            print('Available telscopes: {}'.format(event.dl0.tels_with_data))

            hillas_dict = {}
            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                    tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                    tel_theta[tel_id] = (np.pi/2-event.mc.tel[tel_id].altitude_raw)*u.rad

                if args.photon:
                    pmt_signal = event.mc.tel[tel_id].photo_electron_image
                    new_geom = cam_geom[tel_id]
                else:
                    if cam_geom[tel_id].cam_id == "ASTRI":
                        cal_signal = apply_mc_calibration_ASTRI(
                                        event.r0.tel[tel_id].adc_sums,
                                        event.mc.tel[tel_id].dc_to_pe,
                                        event.mc.tel[tel_id].pedestal)
                    else:
                        cal_signal = apply_mc_calibration(
                            event.r0.tel[tel_id].adc_sums[0],
                            event.mc.tel[tel_id].dc_to_pe[0],
                            event.mc.tel[tel_id].pedestal[0])

                    Imagecutflow.count("calibration")

                    try:
                        pmt_signal, new_geom = \
                            Cleaner.clean(cal_signal, cam_geom[tel_id])
                    except (FileNotFoundError, EdgeEventException) as e:
                        continue
                # end if args.photons

                if Imagecutflow.cut(min_charge_string, np.sum(pmt_signal)):
                    continue

                try:
                    h = hillas_parameters(new_geom.pix_x,
                                          new_geom.pix_y,
                                          pmt_signal)
                    if h.length > 0 and h.width > 0:
                        hillas_dict[tel_id] = h
                except HillasParameterizationError as e:
                    print(e)
                    continue

                Imagecutflow.count("Hillas")

            if Eventcutflow.cut("min2Images", len(hillas_dict)):
                continue

            fit.get_great_circles(hillas_dict, event.inst, *tel_orientation)

            Eventcutflow.count("GreatCircles")

            shower = event.mc
            shower_org = linalg.set_phi_theta(shower.az, 90.*u.deg-shower.alt)

            shower_core = convert_astropy_array([shower.core_x, shower.core_y])

            try:
                fit_position, err_est_dist = fit.fit_core_crosses()
            except Exception as e:
                print([c.norm for c in fit.circles.values()])
                raise e
            if Eventcutflow.cut("nan pos", fit_position):
                continue

            fit_origin = fit.fit_origin_crosses()[0]
            if Eventcutflow.cut("nan dir", fit_origin):
                continue

            xi = linalg.angle(fit_origin, shower_org).to(angle_unit)
            diff = linalg.length(fit_position[:2]-shower_core)

            try:
                # store the reconstruction data in the PyTable
                reco_event["NTels_trigg"] = len(event.dl0.tels_with_data)
                reco_event["NTels_clean"] = len(fit.circles)
                reco_event["EnMC"] = event.mc.energy / energy_unit
                reco_event["xi"] = xi / angle_unit
                reco_event["DeltaR"] = diff / dist_unit
                reco_event["ErrEstPos"] = err_est_dist / dist_unit
                reco_event.append()
                reco_table.flush()

                # print some performance
                print()
                print("xi = {:4.3f}".format(xi))
                print("pos = {:4.3f}".format(diff))
                print("err_est_dist: {:4.3f}".format(err_est_dist))

                print()
                print("xi res (68-percentile) = {:4.3f} {}"
                      .format(np.percentile(reco_table.cols.xi, 68), angle_unit))
                print("core res (68-percentile) = {:4.3f} {}"
                      .format(np.percentile(reco_table.cols.DeltaR, 68), dist_unit))
                print()
            except:
                pass

            # this plots
            # • the MC shower core
            # • the reconstructed shower core
            # • the used telescopes
            # • and the trace of the Hillas plane on the ground
            if False:
                plt.figure()
                for c in fit.circles.values():
                    plt.scatter(c.pos[0], c.pos[1], c="g", s=c.weight)
                    plt.plot([c.pos[0].value-500*c.norm[1], c.pos[0].value+500*c.norm[1]],
                             [c.pos[1].value+500*c.norm[0], c.pos[1].value-500*c.norm[0]])
                plt.scatter(*fit_position[:2], c="r")
                plt.scatter(*shower_core[:2], c="b")
                plt.xlim(-400, 400)
                plt.ylim(-400, 400)
                plt.show()

            if signal_handler.stop: break
        if signal_handler.stop: break

    # print the cutflows for telescopes and camera images
    print()
    Eventcutflow("min2Tels")
    print()
    Imagecutflow(sort_column=1)

    # if we don't want to plot anything, we can exit now
    if not args.plot:
        exit(0)

    figure = plt.figure()
    xi_edges = np.linspace(0, 5, 20)
    plt.hist(reco_table.cols.xi, bins=xi_edges, log=True)
    plt.xlabel(r"$\xi$ / deg")
    if args.write:
        save_fig('{}/reco_xi_{}'.format(args.plots_dir, args.mode))
    plt.pause(.1)

    # convert the xi-list into a dict with the number of used telescopes as keys
    xi_vs_tel = {}
    for xi, ntel in zip(reco_table.cols.xi, reco_table.cols.NTels_clean):
        if ntel not in xi_vs_tel:
            xi_vs_tel[ntel] = [xi]
        else:
            xi_vs_tel[ntel].append(xi)

    print(args.mode)
    for ntel, xis in xi_vs_tel.items():
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

from sys import exit, path
from os.path import expandvars
import numpy as np

from glob import glob

import matplotlib.pyplot as plt
from matplotlib import cm

from ctapipe.utils import linalg

from ctapipe.visualization import CameraDisplay

import seaborn as sns

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.utils.linalg import get_phi_theta, set_phi_theta, angle, length
from ctapipe.utils.CutFlow import CutFlow

from ctapipe.image.hillas import HillasParameterizationError, \
                                 hillas_parameters_4 as hillas_parameters

from ctapipe.reco.HillasReconstructor import \
    HillasReconstructor, TooFewTelescopes
from ctapipe.coordinates.coordinate_transformations import *

from modules.ImageCleaning import ImageCleaner, EdgeEvent
from modules.prepare_event import EventPreparer as EP

from ctapipe.calib import CameraCalibrator

from helper_functions import *

from astropy import units as u
from astropy.table import Table, vstack, hstack
performance_table = Table(names=("Eps_w", "Eps_t",
                                 "alpha_w", "alpha_t",
                                 "hill_width_w", "hill_length_w",
                                 "hill_width_t", "hill_length_t",
                                 "sig_w", "sig_t", "sig_p",
                                 "Event_id", "Tel_id", "N_Tels"))


# your favourite units here
energy_unit = u.GeV
angle_unit = u.deg
dist_unit = u.m


if __name__ == '__main__':
    az_deg = 1

    parser = make_argparser()
    parser.add_argument('--plot_c',  action='store_true',
                        help="plot camera-wise displays")
    parser.add_argument('--add_offset', action='store_true',
                        help="adds a 15 PE offset to all pixels to supress 'Nbr < 0' "
                             "warnings from mrfilter")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--proton',  action='store_true',
                       help="do protons instead of gammas")
    group.add_argument('--electron',  action='store_true',
                       help="do electrons instead of gammas")

    args = parser.parse_args()

    if args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
    elif args.electron:
        filenamelist = glob("{}/electron/*gz".format(args.indir))
    else:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))

    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # pass in config and self if part of a Tool
    calib = CameraCalibrator(None, None)

    # use this in the selection of the gain channels
    np_true_false = np.array([[True], [False]])

    island_cleaning = True
    skip_edge_events = args.skip_edge_events
    Cleaner = {"w": ImageCleaner(mode="wave", cutflow=Imagecutflow,
                                 skip_edge_events=skip_edge_events,
                                 island_cleaning=island_cleaning,
                                 wavelet_options=args.raw),
               "t": ImageCleaner(mode="tail", cutflow=Imagecutflow,
                                 skip_edge_events=skip_edge_events,
                                 island_cleaning=island_cleaning)
               }

    # simple hillas-based shower reco
    fit = HillasReconstructor()

    signal_handler = SignalHandler()
    if args.plot_c:
        signal.signal(signal.SIGINT, signal_handler.stop_drawing)
    else:
        signal.signal(signal.SIGINT, signal_handler)

    # keeping track of the hit distribution transverse to the shower axis on the camera
    # for different energy bins
    from modules.Histogram import nDHistogram
    pe_vs_dp = {'p': {}, 'w': {}, 't': {}}
    for k in pe_vs_dp.keys():
        pe_vs_dp[k] = nDHistogram(
                    bin_edges=[np.arange(6),
                               np.linspace(-.1, .1, 42)*u.m],
                    labels=["log10(signal)", "Delta P"])

    allowed_tels = None
    allowed_tels = prod3b_tel_ids("LSTCam")
    allowed_tels = prod3b_tel_ids("ASTRICam")
    allowed_tels = prod3b_tel_ids("CHEC")
    for filename in sorted(filenamelist)[:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            print()
            print('Available telscopes: {}'.format(event.dl0.tels_with_data))

            # getting the MC shower info
            shower = event.mc
            shower_org = linalg.set_phi_theta(shower.az, 90.*u.deg-shower.alt)
            shower_org = linalg.set_phi_theta(90*u.deg-shower.az, 90.*u.deg-shower.alt)

            org_alt = u.Quantity(shower.alt).to(u.deg)
            org_az = u.Quantity(shower.az).to(u.deg)
            if org_az > 180*u.deg:
                org_az -= 360*u.deg

            org_the = alt_to_theta(org_alt)
            org_phi = az_to_phi(org_az)
            shower_org = linalg.set_phi_theta(org_phi, org_the)

            # calibrate the event
            calib.calibrate(event)

            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                pmt_signal_p = event.mc.tel[tel_id].photo_electron_image

                # getting camera geometry
                camera = event.inst.subarray.tel[tel_id].camera

                if tel_id not in tel_phi:
                    tel_phi[tel_id] = az_to_phi(event.mc.tel[tel_id].azimuth_raw * u.rad)
                    tel_theta[tel_id] = \
                        alt_to_theta(event.mc.tel[tel_id].altitude_raw*u.rad)

                pmt_signal = event.dl1.tel[tel_id].image
                pmt_signal = EP.pick_gain_channel(pmt_signal, camera.cam_id)

                # now cleaning the image with wavelet and tail cuts
                try:
                    pmt_signal_w, new_geom_w = \
                        Cleaner['w'].clean(pmt_signal.copy(), camera)
                    pmt_signal_t, new_geom_t = \
                        Cleaner['t'].clean(pmt_signal.copy(), camera)
                    geom = {'w': new_geom_w, 't': new_geom_t, 'p': camera}
                except (FileNotFoundError, EdgeEvent) as e:
                    print(e)
                    continue

                '''
                do the hillas parametrisation of the two cleaned images '''
                try:
                    hillas = {}
                    # hillas['p'] = hillas_parameters(camera.pix_x,
                    #                                 camera.pix_y,
                    #                                 pmt_signal_p)
                    hillas['w'] = hillas_parameters(new_geom_w,
                                                    pmt_signal_w)
                    hillas['t'] = hillas_parameters(new_geom_t,
                                                    pmt_signal_t)
                except HillasParameterizationError as e:
                    print(e)
                    continue

                Imagecutflow.count("Hillas")

                '''
                get some more parameters and put them in an astropy.table.Table '''
                sum_p = np.sum(pmt_signal_p)
                sum_w = np.sum(pmt_signal_w)
                sum_t = np.sum(pmt_signal_t)

                Epsilon_intensity_w = abs(sum_w - sum_p) / sum_p
                Epsilon_intensity_t = abs(sum_t - sum_p) / sum_p

                alpha = {}
                length = {}
                width = {}
                for k, h in hillas.items():

                    fit.get_great_circles({tel_id: h},
                                          event.inst.subarray, tel_phi, tel_theta)
                    c = fit.circles[tel_id]

                    alpha[k] = abs((angle(c.norm, shower_org)*u.rad) - 90*u.deg).to(u.deg)
                    length[k] = h.length
                    width[k] = h.width

                for k, signal in {  # 'p': pmt_signal_p,
                                  'w': pmt_signal_w}.items():

                    h = hillas[k]

                    p1_x = h.cen_x
                    p1_y = h.cen_y
                    p2_x = p1_x + h.length*np.cos(h.psi)
                    p2_y = p1_y + h.length*np.sin(h.psi)

                    T = linalg.normalise(np.array([(p1_x-p2_x)/u.m, (p1_y-p2_y)/u.m]))

                    x = geom[k].pix_x
                    y = geom[k].pix_y

                    D = [p1_x-x, p1_y-y]

                    dl = D[0]*T[0] + D[1]*T[1]
                    dp = D[0]*T[1] - D[1]*T[0]

                    for pe, pp in zip(signal[abs(dl) > 1*hillas[k].length],
                                      dp[abs(dl) > 1*hillas[k].length]):

                        pe_vs_dp[k].fill([np.log10(sum_p), pp], pe)

                '''
                do some plotting '''
                if args.plot_c and signal_handler.draw:
                    fig = plt.figure(figsize=(15, 5))

                    # ax1 = fig.add_subplot(221)
                    # disp1 = CameraDisplay(camera,
                    #                       image=pmt_signal_p,
                    #                       ax=ax1)
                    # disp1.cmap = plt.cm.inferno
                    # disp1.add_colorbar()
                    # # disp1.overlay_moments(hillas['p'], color='seagreen', linewidth=3)
                    # # plt.title("PE image ; alpha = {:4.3f}".format(alpha['p']))

                    # ax2 = fig.add_subplot(222)
                    ax2 = fig.add_subplot(131)
                    disp2 = CameraDisplay(camera,
                                          image=pmt_signal,
                                          ax=ax2)
                    disp2.cmap = plt.cm.inferno
                    disp2.add_colorbar(label="signal")
                    plt.title("calibrated noisy image")

                    # ax3 = fig.add_subplot(223)
                    ax3 = fig.add_subplot(132)
                    disp3 = CameraDisplay(new_geom_t,
                                          image=np.sqrt(pmt_signal_t),
                                          ax=ax3)
                    disp3.cmap = plt.cm.inferno
                    disp3.add_colorbar(label="sqrt(signal)")
                    disp3.overlay_moments(hillas['t'], color='seagreen', linewidth=3)
                    plt.title("tailcut cleaned ({},{}) ; alpha = {:4.3f}"
                              .format(Cleaner['t'].tail_thresholds[new_geom_t.cam_id][0],
                                      Cleaner['t'].tail_thresholds[new_geom_t.cam_id][1],
                                      alpha['t']))

                    # ax4 = fig.add_subplot(224)
                    ax4 = fig.add_subplot(133)
                    disp4 = CameraDisplay(new_geom_w,
                                          image=np.sqrt(
                                              np.sum(pmt_signal_w, axis=1)
                                              if pmt_signal_w.shape[-1] == 25
                                              else pmt_signal_w),
                                          ax=ax4)
                    hw = hillas['w']
                    disp4.cmap = plt.cm.inferno
                    disp4.add_colorbar(label="sqrt(signal)")
                    disp4.overlay_moments(hillas['w'], color='seagreen', linewidth=3)
                    plt.title("wavelet cleaned ; alpha = {:4.3f}".format(alpha['w']))
                    plt.suptitle("Camera {}".format(tel_id))
                    plt.show()

                # if there is any nan values, skip
                if np.isnan([Epsilon_intensity_w,
                             Epsilon_intensity_t,
                             alpha['w'].value,
                             alpha['t'].value,
                             width['w'].value, length['w'].value,
                             width['t'].value, length['t'].value]).any():
                    continue

                '''
                now fill the table '''
                performance_table.add_row([Epsilon_intensity_w, Epsilon_intensity_t,
                                           alpha['w'], alpha['t'],
                                           width['w'], length['w'],
                                           width['t'], length['t'],
                                           sum_w, sum_t, sum_p,
                                           event.dl0.event_id, tel_id,
                                           len(event.dl0.tels_with_data)])

            if not len(performance_table):
                continue

            '''
            determine and print the 68-percentile of the hillas ellipsis tilt error
            of the two cleaning methods '''
            alphas_w = performance_table["alpha_w"]
            alphas_t = performance_table["alpha_t"]
            print()
            print("alpha_w res (68-percentile) = {}".format(np.percentile(alphas_w, 68)))
            print("alpha_t res (68-percentile) = {}".format(np.percentile(alphas_t, 68)))

            if signal_handler.stop: break
        if signal_handler.stop: break

    # print the cutflow
    print()
    Imagecutflow()

    # if we don't want to plot anything, we can exit now
    if not args.plot:
        exit(0)

    tab1 = performance_table["sig_p", "alpha_t"]
    tab2 = performance_table["sig_p", "alpha_w"]
    tab1.rename_column("alpha_t", "alpha")
    tab2.rename_column("alpha_w", "alpha")

    data = vstack([tab1, tab2])

    npe_centres = np.linspace(1, 6, 21)
    npe_edges = (npe_centres[1:]+npe_centres[:-1])/2.
    data["log10(sig_p)"] = npe_centres[
            np.clip(
                np.digitize(np.log10(data["sig_p"]), npe_edges)-1,
                0, len(npe_edges)-1)
            ]

    data["log10(alpha)"] = np.log10(data["alpha"])
    data["mode"] = ['tail']*len(tab1) + ['wave']*len(tab2)

    # sns.set_context("paper", rc={"lines.linewidth": 10})
    sns.violinplot(x="log10(sig_p)", y="log10(alpha)", hue="mode", data=data.to_pandas(),
                   palette="Set2", inner="quartiles", split=True)
    # sns.swarmplot(x="log10(sig_p)", y="log10(alpha)", hue="mode", data=data.to_pandas(),
    #               palette="Set1", split=False)
    plt.pause(.1)

    # print how many telescopes participated in each log-signal bin
    sig_p = performance_table["sig_p"]
    for k in ['p', 't', 'w']:
        print("type:", k)
        for log_sig_l, log_sig_h in zip(pe_vs_dp[k].bin_edges[0][:-1],
                                        pe_vs_dp[k].bin_edges[0][1:]):
            print("signal range: {} -- {}\t number of tels {}".format(
                log_sig_l, log_sig_h,
                len(sig_p[(sig_p < 10**(log_sig_h)) & (sig_p > 10**log_sig_l)])))
        print()

    if args.write:
        performance_table.write("Eps_int_comparison.fits", overwrite=True)

    if args.verbose:
        pe_vs_dp_p = pe_vs_dp['p'].normalise()
        pe_vs_dp_w = pe_vs_dp['w'].normalise()
        plt.figure()
        plt.subplot(121)
        shape = pe_vs_dp_p.data[1:-1, 1:-1].shape
        norm = np.repeat(pe_vs_dp_p.data[1:-1, 1:-1].max(axis=1), shape[1]).reshape(shape)
        plt.imshow(pe_vs_dp_p.data[1:-1, 1:-1] / norm,
                   extent=(pe_vs_dp_p.bin_edges[1][0].value,
                           pe_vs_dp_p.bin_edges[1][-1].value,
                           pe_vs_dp_p.bin_edges[0][0],
                           pe_vs_dp_p.bin_edges[0][-1]),
                   cmap=plt.cm.inferno,
                   origin='lower',
                   aspect='auto',
                   interpolation='none')
        plt.title("photo electrons")
        plt.colorbar()

        plt.subplot(122)
        shape = pe_vs_dp_w.data[1:-1, 1:-1].shape
        norm = np.repeat(pe_vs_dp_w.data[1:-1, 1:-1].max(axis=1), shape[1]).reshape(shape)
        plt.imshow(pe_vs_dp_w.data[1:-1, 1:-1] / norm,
                   extent=(pe_vs_dp_p.bin_edges[1][0].value,
                           pe_vs_dp_p.bin_edges[1][-1].value,
                           pe_vs_dp_p.bin_edges[0][0],
                           pe_vs_dp_p.bin_edges[0][-1]),
                   cmap=plt.cm.inferno,
                   origin='lower',
                   aspect='auto',
                   interpolation='none')
        plt.title("wavelet cleaned")
        plt.colorbar()
        plt.pause(.1)

        for pe_bin in []:  # [2, 3, 4, 5]:
            fig = plt.figure()
            if np.sum(pe_vs_dp_w.norm[pe_bin][1:-1]) > 0:
                plt.style.use('t_slides')
                bin_centres = (pe_vs_dp_p.bin_edges[1][1:]+pe_vs_dp_p.bin_edges[1][:-1])/2

                plt.suptitle("total signal: 10^{} to 10^{}"
                             .format(pe_vs_dp_p.bin_edges[0][pe_bin-1],
                                     pe_vs_dp_p.bin_edges[0][pe_bin]))

                plt.subplot(131)
                plt.plot(bin_centres, pe_vs_dp_w.norm[pe_bin][1:-1])
                plt.title("hit pixel")
                plt.xlabel("perpendicular offset / m")
                plt.ylabel("number of hit pmt")

            if np.sum(pe_vs_dp_p.data[pe_bin][1:-1]) > 0:
                plt.subplot(132)
                plt.semilogy(bin_centres, pe_vs_dp_p.data[pe_bin][1:-1], 'b', label='PE')
                plt.semilogy(bin_centres, pe_vs_dp_w.data[pe_bin][1:-1], 'r', label='wave')
                plt.title("PMT signal")
                plt.xlabel("perpendicular offset / m")
                plt.ylabel("average pmt signal")
                plt.legend()

            if np.sum(pe_vs_dp_w.data[pe_bin][1:-1]) > 0:
                plt.subplot(133)

                ratio = np.zeros_like(pe_vs_dp_w.data[pe_bin][1:-1])
                ratio[pe_vs_dp_p.data[pe_bin][1:-1] > 0] = \
                    pe_vs_dp_w.data[pe_bin][1:-1][pe_vs_dp_p.data[pe_bin][1:-1] > 0] / \
                    pe_vs_dp_p.data[pe_bin][1:-1][pe_vs_dp_p.data[pe_bin][1:-1] > 0]

                plt.plot(bin_centres, ratio)
                plt.title("signal ratio")
                plt.xlabel("perpendicular offset / m")
                plt.ylabel("wave signal / real signal")

                plt.pause(.1)

    npe_edges = np.linspace(1, 6, 21)
    size_edges = npe_edges
    lovw_edges = np.linspace(0, 3, 16)

    sig_p = performance_table["sig_p"]

    for k in Cleaner.keys():

        Epsilon_2 = performance_table["Eps_{}".format(k)]
        hillas_tilt = performance_table["alpha_{}".format(k)] * u.deg
        hillas_length = performance_table["hill_length_{}".format(k)]
        hillas_width = performance_table["hill_width_{}".format(k)]
        sig = performance_table["sig_{}".format(k)]

        mode = "wave" if k == "w" else "tail"

        plt.figure()
        plot_hex_and_violin(
                np.log10(Epsilon_2),
                np.log10(sig_p),
                bin_edges=None,
                ylabel="log10(NPE)",
                xlabel="log10(Epsilon 2)",
                zlabel="log10(counts)",
                bins='log',
                extent=(-.5, 0, 1.5, 5),
                do_violin=False)
        plt.grid()
        plt.suptitle(mode)
        plt.tight_layout()
        plt.pause(.1)

        # plot the angular error of the hillas ellipsis vs the number of photo electrons
        plt.figure()
        plot_hex_and_violin(np.log10(sig_p),
                            np.log10(hillas_tilt/angle_unit),
                            npe_edges,
                            extent=[1, 4, -5, 1],
                            xlabel="log10(number of photo electrons)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.grid()
        plt.suptitle(mode)
        plt.tight_layout()
        if args.write:
            save_fig("plots/alpha_vs_photoelecrons_{}".format(mode))
        plt.pause(.1)

        # plot the angular error of the hillas ellipsis vs the measured camera signal
        plt.figure()
        plot_hex_and_violin(np.log10(sig),
                            np.log10(hillas_tilt/angle_unit),
                            size_edges,
                            extent=[1, 4, -5, 1],
                            xlabel="log10(signal size)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.grid()
        plt.suptitle(mode)
        plt.tight_layout()
        if args.write:
            save_fig("plots/alpha_vs_signal_{}".format(mode))
        plt.pause(.1)

        # plot the angular error of the hillas ellipsis vs the length/width ratio
        # plt.figure()
        # plot_hex_and_violin(np.log10(hillas_length/hillas_width),
        #                     np.log10(hillas_tilt/angle_unit),
        #                     lovw_edges,
        #                     extent=[0, 2, -4.5, 1],
        #                     xlabel="log10(length/width)",
        #                     ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        # plt.grid()
        # plt.suptitle(mode)
        # plt.tight_layout()
        # if args.write:
        #     save_fig("plots/alpha_vs_lenOVwidth_{}".format(mode))
        # plt.pause(.1)

    plt.show()

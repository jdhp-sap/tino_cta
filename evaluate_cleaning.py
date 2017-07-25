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

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.linalg import get_phi_theta, set_phi_theta, angle, length

from ctapipe.image.hillas import HillasParameterizationError, \
                                 hillas_parameters_4 as hillas_parameters

try:
    from ctapipe.reco.FitGammaHillas import \
        FitGammaHillas as HillasReconstructor, TooFewTelescopesException
except ImportError:
    from ctapipe.reco.HillasReconstructor import \
        HillasReconstructor, TooFewTelescopesException

from modules.ImageCleaning import ImageCleaner, EdgeEventException
from modules.CutFlow import CutFlow

from modules.reSampling import resample_hex_to_rect, \
                               make_qr_to_pix_id_map, \
                               qr_to_pix_id_map_tel_map

from ctapipe.calib import CameraCalibrator

from helper_functions import *

from astropy import units as u
from astropy.table import Table, vstack, hstack
performance_table = Table(names=("Eps_w", "Eps_t",
                                 "alpha_w", "alpha_t", "alpha_s",
                                 "hill_width_w", "hill_length_w",
                                 "hill_width_t", "hill_length_t",
                                 "sig_w", "sig_t", "sig_p",
                                 "Event_id", "Tel_id", "N_Tels"))


# your favourite units here
energy_unit = u.GeV
angle_unit = u.deg
dist_unit = u.m


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")
    parser.add_argument('--plot_c',  action='store_true',
                        help="plot camera-wise displays")
    parser.add_argument('--add_offset', action='store_true',
                        help="adds a 15 PE offset to all pixels to supress 'Nbr < 0' "
                             "warnings from mrfilter")

    args = parser.parse_args()

    if args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
    else:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))

    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # pass in config and self if part of a Tool
    calib = CameraCalibrator(None, None)

    # use this in the selection of the gain channels
    np_true_false = np.array([[True], [False]])

    island_cleaning = False
    skip_edge_events = False  # args.skip_edge_events
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
    # allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = range(34, 40)  # use the array of FlashCams instead
    for filename in sorted(filenamelist)[:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            print()
            print('Scanning input file... count = {}'.format(event.count))
            print('Event ID: {}'.format(event.dl0.event_id))
            print('Available telscopes: {}'.format(event.dl0.tels_with_data))

            # getting the MC shower info
            shower = event.mc
            shower_org = linalg.set_phi_theta(shower.az, 90.*u.deg-shower.alt)

            # calibrate the event
            calib.calibrate(event)

            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                pmt_signal_p = event.mc.tel[tel_id].photo_electron_image

                # guessing camera geometry
                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                    tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                    tel_theta[tel_id] = (np.pi/2-event.mc.tel[tel_id].altitude_raw)*u.rad

                pmt_signal = event.dl1.tel[tel_id].image
                if pmt_signal.shape[0] > 1:
                    pick = (pmt_signal > 14).any(axis=0) != np_true_false
                    pmt_signal = pmt_signal.T[pick.T]
                else:
                    pmt_signal = pmt_signal.ravel()

                #
                # resampling of the hex grid into sqare grid
                #pix_x, pix_y = cam_geom[tel_id].pix_x, cam_geom[tel_id].pix_y
                #hex_size = (cam_geom[tel_id].pix_area[0] *
                            #(2/3**0.5)**3)**0.5 / 2
                #nx, ny = 150, 150

                #if cam_geom[tel_id].cam_id not in qr_to_pix_id_map_tel_map:
                    #qr_to_pix_id_map = make_qr_to_pix_id_map(pix_x.value,
                                                             #pix_y.value,
                                                             #hex_size.value)
                    #qr_to_pix_id_map_tel_map[cam_geom[tel_id].cam_id] = qr_to_pix_id_map

                #resample_x, resample_y, resample_img = \
                    #resample_hex_to_rect(pmt_signal, geom=cam_geom[tel_id],
                                         #nx=nx, ny=ny)



                #min_x = np.min(pix_x)
                #max_x = np.max(pix_x)
                #min_y = np.min(pix_y)
                #max_y = np.max(pix_y)

                #lim_x, lim_y = 0.57, 1.04
                #trimmed_x   = resample_x  [(np.abs(resample_x) < lim_x) &
                                           #(np.abs(resample_y) < lim_y)]
                #trimmed_y   = resample_y  [(np.abs(resample_x) < lim_x) &
                                           #(np.abs(resample_y) < lim_y)]
                #trimmed_img = resample_img[(np.abs(resample_x) < lim_x) &
                                           #(np.abs(resample_y) < lim_y)]

                #from datapipe.denoising.wavelets_mrfilter import WaveletTransform
                #wavelet_transform = WaveletTransform()
                #rect_img = np.histogram2d(trimmed_x, trimmed_y, weights=trimmed_img,
                                          #bins=(np.linspace(min_x, max_x, nx),
                                                #np.linspace(min_y, max_y, ny)))[0].T

                #rect_x, rect_y = np.meshgrid(np.linspace(min_x, max_x, nx-1),
                                             #np.linspace(min_y, max_y, ny-1))

                #clean_rect_img = wavelet_transform(
                    #rect_img,
                    #raw_option_string=args.raw)

                #clean_rect_img = kill_isolpix(clean_rect_img, threshold=2)

                #from copy import copy
                #trim_geom = copy(cam_geom[tel_id])
                #trim_geom.pix_x = rect_x.ravel()
                #trim_geom.pix_y = rect_y.ravel()
                #trim_geom.mask = np.ones_like(clean_rect_img.ravel())
                #trim_geom.pixel_shape = "rectangular"
                #trim_geom.cam_rotation = 0 * u.deg
                #trim_geom.pix_area = np.ones_like(clean_rect_img.ravel()) * .6 *\
                    #abs(trimmed_x[0]-trimmed_x[1]) * \
                    #abs(trimmed_y[0]-trimmed_y[ny])*u.m*u.m

                #fig = plt.figure()
                #ax1 = fig.add_subplot(221)
                #disp1 = CameraDisplay(cam_geom[tel_id],
                                      #image=pmt_signal_p,
                                      #ax=ax1)
                #disp1.cmap = plt.cm.hot
                #disp1.add_colorbar()
                #plt.title("PE image")


                #ax2 = fig.add_subplot(222)
                #disp2 = CameraDisplay(cam_geom[tel_id],
                                    #image=pmt_signal,
                                    #ax=ax2)
                #disp2.cmap = plt.cm.hot
                #disp2.add_colorbar()
                #plt.title("calibrated noisy image")

                #ax3 = fig.add_subplot(223)


                #plt.hist2d(trimmed_x, trimmed_y, weights=trimmed_img,
                           #bins=(np.linspace(min_x, max_x, nx),
                                 #np.linspace(min_y, max_y, ny)),
                           #cmap=plt.cm.hot
                           #)
                #ax3.set_aspect('equal')
                #plt.colorbar()
                #plt.title("resampled, trimmed noisy image")

                #ax4 = fig.add_subplot(224)
                #plt.imshow(clean_rect_img, interpolation='none', origin='lower',
                           #cmap=plt.cm.hot)
                #plt.colorbar()
                #plt.title("trimmed \"filtered\" image")

                #plt.show()

                #continue


                # now cleaning the image with wavelet and tail cuts
                try:
                    pmt_signal_w, new_geom_w = \
                        Cleaner['w'].clean(pmt_signal+5
                                           if args.add_offset else pmt_signal,
                                           cam_geom[tel_id])
                    pmt_signal_t, new_geom_t = \
                        Cleaner['t'].clean(pmt_signal.copy(), cam_geom[tel_id])
                    geom = {'w': new_geom_w, 't': new_geom_t, 'p': cam_geom[tel_id]}
                except (FileNotFoundError, EdgeEventException) as e:
                    print(e)
                    continue

                '''
                do the hillas parametrisation of the two cleaned images '''
                try:
                    hillas = {}
                    hillas['p'] = hillas_parameters(cam_geom[tel_id].pix_x,
                                                    cam_geom[tel_id].pix_y,
                                                    pmt_signal_p)
                    hillas['w'] = hillas_parameters(new_geom_w.pix_x,
                                                    new_geom_w.pix_y,
                                                    pmt_signal_w)
                    hillas['t'] = hillas_parameters(new_geom_t.pix_x,
                                                    new_geom_t.pix_y,
                                                    pmt_signal_t)
                    #hillas['s'] = hillas_parameters(trim_geom.pix_x,
                                                    #trim_geom.pix_y,
                                                    #clean_rect_img.ravel())
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
                                          event.inst, tel_phi, tel_theta)
                    c = fit.circles[tel_id]

                    alpha[k] = abs((angle(c.norm, shower_org)*u.rad) - 90*u.deg).to(u.deg)
                    length[k] = h.length
                    width[k] = h.width

                for k, signal in {'p': pmt_signal_p,
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

                    for pe, pp in zip(signal[abs(dl) > 1*hillas['p'].length],
                                      dp[abs(dl) > 1*hillas['p'].length]):

                        pe_vs_dp[k].fill([np.log10(sum_p), pp], pe)

                '''
                do some plotting '''
                if args.plot_c and signal_handler.draw:
                    fig = plt.figure(figsize=(10,10))

                    ax1 = fig.add_subplot(221)
                    disp1 = CameraDisplay(cam_geom[tel_id],
                                          image=pmt_signal_p,
                                          ax=ax1)
                    disp1.cmap = plt.cm.hot
                    disp1.add_colorbar()
                    disp1.overlay_moments(hillas['p'], color='seagreen', linewidth=3)
                    plt.title("PE image ; alpha = {:4.3f}".format(alpha['p']))

                    ax2 = fig.add_subplot(222)
                    disp2 = CameraDisplay(cam_geom[tel_id],
                                          image=pmt_signal,
                                          ax=ax2)
                    disp2.cmap = plt.cm.hot
                    disp2.add_colorbar()
                    plt.title("calibrated noisy image")

                    ax3 = fig.add_subplot(223)
                    #disp3 = CameraDisplay(trim_geom,
                                          #image=clean_rect_img.ravel(),
                                          #ax=ax3)
                    #disp3.overlay_moments(hillas['s'], color='seagreen', linewidth=3)
                    #disp3.cmap = plt.cm.hot
                    #disp3.add_colorbar()
                    #plt.title("wave trimm cleaned ; alpha = {:4.3f}"
                              #.format(alpha['t']))

                    disp3 = CameraDisplay(new_geom_t,
                                          image=np.sqrt(pmt_signal_t),
                                          ax=ax3)
                    disp3.cmap = plt.cm.hot
                    disp3.add_colorbar()
                    disp3.overlay_moments(hillas['t'], color='seagreen', linewidth=3)
                    plt.title("tail cleaned ({},{}) ; alpha = {:4.3f}"
                              .format(Cleaner['t'].tail_thresh_up,
                                      Cleaner['t'].tail_thresh_low,
                                      alpha['t']))

                    ax4 = fig.add_subplot(224)
                    disp4 = CameraDisplay(new_geom_w,
                                          image=np.sqrt(
                                                    np.sum(pmt_signal_w, axis=1)
                                                    if pmt_signal_w.shape[-1] == 25
                                                    else pmt_signal_w),
                                          ax=ax4)
                    disp4.cmap = plt.cm.hot
                    disp4.add_colorbar()
                    disp4.overlay_moments(hillas['w'], color='seagreen', linewidth=3)
                    plt.title("wave slant cleaned ; alpha = {:4.3f}".format(alpha['w']))
                    plt.suptitle("Camera {}".format(tel_id))
                    plt.show()

                '''
                if there is any nan values, skip '''
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
                                           alpha['w'], alpha['t'], -100, #alpha['s'],
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
            alphas_s = performance_table["alpha_s"]
            print()
            print("alpha_w res (68-percentile) = {}".format(np.percentile(alphas_w, 68)))
            print("alpha_t res (68-percentile) = {}".format(np.percentile(alphas_t, 68)))
            print("alpha_s res (68-percentile) = {}".format(np.percentile(alphas_s, 68)))

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
    #sns.swarmplot(x="log10(sig_p)", y="log10(alpha)", hue="mode", data=data.to_pandas(),
                  #palette="Set1", split=False)
    plt.show()

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

        for pe_bin in [2, 3, 4, 5]:
            if np.sum(pe_vs_dp_w.norm[pe_bin][1:-1]) > 0:
                fig = plt.figure()
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

        plot_hex_and_violin(
                np.log10(Epsilon_2),
                np.log10(sig_p),
                bin_edges=None,
                ylabel="log10(NPE)",
                xlabel="log10(Epsilon 2)",
                zlabel="log10(counts)",
                bins='log',
                extent=(-3, 0, 1.5, 5),
                do_violin=False)
        plt.suptitle(mode)
        plt.tight_layout()
        plt.pause(.1)

        '''
        plot the angular error of the hillas ellipsis vs the number of photo electrons '''
        plot_hex_and_violin(np.log10(sig_p),
                            np.log10(hillas_tilt/angle_unit),
                            npe_edges,
                            extent=[0, 5, -5, 1],
                            xlabel="log10(number of photo electrons)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.suptitle(mode)
        plt.tight_layout()
        if args.write:
            save_fig("plots/alpha_vs_photoelecrons_{}".format(mode))
        plt.pause(.1)

        '''
        plot the angular error of the hillas
        ellipsis vs the measured signal on the camera '''
        plot_hex_and_violin(np.log10(sig_p),
                            np.log10(hillas_tilt/angle_unit),
                            size_edges,
                            extent=[0, 5, -5, 1],
                            xlabel="log10(signal size)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.suptitle(mode)
        plt.tight_layout()
        if args.write:
            save_fig("plots/alpha_vs_signal_{}".format(mode))
        plt.pause(.1)

        '''
        plot the angular error of the hillas ellipsis vs the length/width ratio '''
        plot_hex_and_violin(np.log10(hillas_length/hillas_width),
                            np.log10(hillas_tilt/angle_unit),
                            lovw_edges,
                            extent=[0, 2, -4.5, 1],
                            xlabel="log10(length/width)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.suptitle(mode)
        plt.tight_layout()
        if args.write:
            save_fig("plots/alpha_vs_lenOVwidth_{}".format(mode))
        plt.pause(.1)

    plt.show()

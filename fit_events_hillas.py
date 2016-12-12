from sys import exit, path
from os.path import expandvars
import math
import numpy as np

from glob import glob

from bisect import insort

import matplotlib.pyplot as plt
from matplotlib import cm

from astropy import units as u
from astropy.table import Table
eps_table = Table(names=("Eps_w", "Eps_t", "diff_Eps", "sig_w", "sig_t", "sig_p", "Event_id", "Tel_id"))


from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.linalg import get_phi_theta, set_phi_theta, angle, length

from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException


path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe"))

from extract_and_crop_simtel_images import crop_astri_image

from modules.ImageCleaning import ImageCleaner, EdgeEventException
from modules.CutFlow import CutFlow

from helper_functions import *


old = False
# old = True


''' your favourite units here '''
angle_unit  = u.deg
energy_unit = u.GeV
dist_unit   = u.m


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")
    args = parser.parse_args()

    if args.proton:
        filenamelist = glob("{}/proton/*run{}*gz".format(args.indir, args.runnr))
    else:
        filenamelist = glob("{}/gamma/*run{}*gz".format(args.indir, args.runnr))

    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    min_charge = "min_charge {}".format(args.min_charge)
    Imagecutflow.set_cut(min_charge, lambda x: x > args.min_charge)



    # Cleaner = ImageCleaner(mode="none")
    Cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           skip_edge_events=False, island_cleaning=False)

    fit = FitGammaHillas()

    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    NTels = []
    EnMC  = []
    xis1  = []
    xis2  = []
    xisd  = []

    diffs = []

    tel_signal = []
    tel_signal_pe = []
    hillas_tilt = []
    hillas_length = []
    hillas_width = []

    # allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = range(34, 40)  # use the array of FlashCams instead
    for filename in sorted(filenamelist)[:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            Eventcutflow.count("noCuts")

            if len(event.dl0.tels_with_data) < 2:
                continue

            Eventcutflow.count("min2Tels")

            print('Scanning input file... count = {}'.format(event.count))
            print('Event ID: {}'.format(event.dl0.event_id))
            print('Available telscopes: {}'.format(event.dl0.tels_with_data))

            hillas_dict = {}
            # for tel_id, tel in event.mc.tel.items():
                # pmt_signal = tel.photo_electrons

            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                    tel_phi[tel_id] = 180.*u.deg
                    tel_theta[tel_id] = 20.*u.deg

                if cam_geom[tel_id] == "ASTRI":
                    cal_signal = apply_mc_calibration_ASTRI(
                                    event.dl0.tel[tel_id].adc_sums,
                                    event.mc.tel[tel_id].dc_to_pe,
                                    event.mc.tel[tel_id].pedestal)
                else:
                    cal_signal = apply_mc_calibration(
                        event.dl0.tel[tel_id].adc_sums[0],
                        event.mc.tel[tel_id].dc_to_pe[0],
                        event.mc.tel[tel_id].pedestal[0])

                Imagecutflow.count("calibration")

                try:
                    pmt_signal, new_geom = \
                        Cleaner.clean(cal_signal, cam_geom[tel_id],
                                      event.inst.optical_foclen[tel_id])
                    pmt_signal_2 = \
                        Cleaner.clean_tail(cal_signal, cam_geom[tel_id],
                                           event.inst.optical_foclen[tel_id])[0]
                except FileNotFoundError as e:
                    print(e)
                    continue
                except EdgeEventException:
                    continue


                sum_cleaned = np.sum(pmt_signal)
                sum_signal = np.sum(event.mc.tel[tel_id].photo_electron_image)
                Epsilon_intensity = abs(sum_cleaned - sum_signal) / sum_signal

                sum_cleaned_2 = np.sum(pmt_signal_2)
                Epsilon_intensity_2 = abs(sum_cleaned_2 - sum_signal) / sum_signal
                eps_table.add_row([Epsilon_intensity,
                                   Epsilon_intensity_2,
                                   Epsilon_intensity - Epsilon_intensity_2,
                                   sum_cleaned,
                                   sum_cleaned_2,
                                   sum_signal,
                                   event.dl0.event_id,
                                   tel_id])
                continue

                #from ctapipe.visualization import CameraDisplay
                #fig = plt.figure()
                #plt.style.use('seaborn-talk')

                #ax2 = fig.add_subplot(121)
                #disp2 = CameraDisplay(cam_geom[tel_id],
                                      #image=event.dl0.tel[tel_id].adc_sums[0],
                                      #ax=ax2)
                #disp2.cmap = plt.cm.hot
                #disp2.add_colorbar()

                #ax1 = fig.add_subplot(122)
                #disp1 = CameraDisplay(new_geom,
                                      #image=np.sum(pmt_signal, axis=1)
                                      #if pmt_signal.shape[-1] == 25 else pmt_signal,
                                      #ax=ax1)
                #disp1.cmap = plt.cm.hot
                #disp1.add_colorbar()
                #plt.title("Camera {}".format(tel_id))
                #plt.suptitle("{} mode".format(args.mode))
                #plt.show()


                if not Imagecutflow.cut(min_charge, np.sum(pmt_signal)):
                    continue

                try:
                    hillas_dict[tel_id] = hillas_parameters(new_geom.pix_x,
                                                            new_geom.pix_y,
                                                            pmt_signal)[0]
                except HillasParameterizationError as e:
                    print(e)
                    continue

                Imagecutflow.count("Hillas")

            Eventcutflow.count("ImagePreparation")

            fit.get_great_circles(hillas_dict, event.inst, *tel_orientation)

            Eventcutflow.count("GreatCircles")

            shower = event.mc
            # corsika measures azimuth the other way around, using phi=-az
            shower_dir = set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
            # shower direction is downwards, shower origin up
            shower_org = -shower_dir

            for k in fit.circles.keys():
                c = fit.circles[k]
                h = hillas_dict[k]
                tel_signal.append(h.size)
                tel_signal_pe.append(np.sum(event.mc.tel[k].photo_electron_image))
                hillas_tilt.append(abs((angle(c.norm, shower_org)*u.rad) - 90*u.deg))
                hillas_length.append(h.length * u.m)
                hillas_width.append(h.width * u.m)

            shower_core = np.array([shower.core_x.value,
                                    shower.core_y.value])*u.m

            try:
                result1 = fit.fit_origin_crosses()[0]
                result2 = fit.fit_origin_minimise(result1)

                seed = np.mean([event.inst.tel_pos[tel_id]
                               for tel_id in fit.circles.keys()], axis=0)[:2]*u.m
                seed = [0, 0]*u.m
                pos_fit = fit.fit_core(seed)

            except TooFewTelescopesException as e:
                print(e)
                continue

            Eventcutflow.count("Reco")

            xi1 = angle(result1, shower_org).to(angle_unit)
            xi2 = angle(result2, shower_org).to(angle_unit)

            print()
            print("xi1 = {}".format(xi1))
            print("xi2 = {}".format(xi2))
            print("x1-xi2 = {}".format(xi1-xi2))

            insort(xis1, xi1)
            insort(xis2, xi2)
            insort(xisd, xi1-xi2)

            NEvents = len(xis2)
            print()
            print("xi1 res (68-percentile) = {}"
                  .format(xis1[int(NEvents*.68)]))
            print("xi2 res (68-percentile) = {}"
                  .format(xis2[int(NEvents*.68)]))
            print("median difference = {}".format(xisd[NEvents//2]))
            print()

            diff = length(pos_fit[:2]-shower_core)
            print("reco = ", diff)
            insort(diffs, diff)
            print("core res (68-percentile) = {}"
                  .format(diffs[int(len(diffs)*.68)]))
            print()
            print("Events:", NEvents)
            print()

            '''
            save number of telescopes and MC energy for this event '''
            NTels.append(len(fit.circles))
            EnMC.append(event.mc.energy)

            if signal_handler.stop: break
        if signal_handler.stop: break

    '''
    print the cutflows for telescopes and camera images '''
    print()
    Eventcutflow("min2Tels")
    print()
    Imagecutflow()

    print(eps_table)

    eps_table.write("Eps_int_comparison.fits", overwrite=True)

    plot_hex_and_violin(
            np.log10(eps_table["Eps_w"]),
            np.log10(eps_table["sig_p"]),
            None,
            ylabel="log10(NPE)",
            xlabel="log10(Eps2 wave)",
            zlabel="log10(counts)",
            bins='log',
            extent=(-3, 0, 1.5, 5),
            do_violin=False)
    plt.pause(.1)
    plot_hex_and_violin(
            np.log10(eps_table["Eps_t"]),
            np.log10(eps_table["sig_p"]),
            None,
            ylabel="log10(NPE)",
            xlabel="log10(Eps2 tail)",
            zlabel="log10(counts)",
            bins='log',
            extent=(-3, 0, 1.5, 5),
            do_violin=False)
    plt.show()

    '''
    if we don't want to plot anything, we can exit now '''
    if not args.plot:
        exit(0)

    hillas_tilt = convert_astropy_array(hillas_tilt)
    hillas_length = convert_astropy_array(hillas_length)
    hillas_width = convert_astropy_array(hillas_width)

    '''
    plot the angular error of the hillas ellipsis vs the number of photo electrons '''
    npe_edges = np.linspace(1, 6, 21)
    plot_hex_and_violin(np.log10(tel_signal_pe),
                        np.log10(hillas_tilt/angle_unit),
                        npe_edges,
                        extent=[0, 5, -5, 1],
                        xlabel="log10(number of photo electrons)",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)
    if args.write:
        save_fig("plots/alpha_vs_photoelecrons_{}".format(args.mode))
    plt.pause(.1)

    '''
    plot the angular error of the hillas ellipsis vs the measured signal on the camera '''
    size_edges = np.linspace(1, 6, 21)
    plot_hex_and_violin(np.log10(tel_signal),
                        np.log10(hillas_tilt/angle_unit),
                        size_edges,
                        extent=[0, 5, -5, 1],
                        xlabel="log10(signal size)",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)
    if args.write:
        save_fig("plots/alpha_vs_signal_{}".format(args.mode))
    plt.pause(.1)

    '''
    plot the angular error of the hillas ellipsis vs the length/width ratio '''
    lovw_edges = np.linspace(0, 3, 16)
    plot_hex_and_violin(np.log10(hillas_length/hillas_width),
                        np.log10(hillas_tilt/angle_unit),
                        lovw_edges,
                        extent=[0, 2, -4.5, 1],
                        xlabel="log10(length/width)",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)
    if args.write:
        save_fig("plots/alpha_vs_lenOVwidth_{}".format(args.mode))
    plt.show()


    # xis1 = convert_astropy_array(xis1)
    xis2 = convert_astropy_array(xis2)
    # xisb = convert_astropy_array(xisb)

    xis = xis2

    figure = plt.figure()
    plt.hist(xis, bins=np.linspace(0, 25, 50), log=True)
    plt.xlabel(r"$\xi$ / deg")
    if args.write:
        save_fig('plots/xi_{}'.format(args.mode), draw_rectangles=True)
    plt.pause(.1)

    #figure = plt.figure()
    #plt.hist(np.log10(xis2/angle_unit), bins=np.linspace(-3, 1, 50))
    #plt.xlabel(r"log($\xi_2$ / deg)")
    #if args.write:
        #save_fig('plots/'+args.mode+'_xi2', draw_rectangles=True)
    #plt.pause(.1)

    #figure = plt.figure()
    #plt.hist(np.log10(xisb/angle_unit), bins=np.linspace(-.5, .5, 50))
    #plt.xlabel(r"$\log(\xi_\mathrm{best} / \deg)$")
    #if args.write:
        #save_fig('plots/'+args.mode+'_xi_best', draw_rectangles=True)
    #plt.pause(.1)

    #figure = plt.figure()
    #plt.hist((xis1-xis2), bins=np.linspace(-.65, .65, 13), log=True)
    #plt.xlabel(r"$(\xi_1 - \xi_2) / \deg)$")
    #if args.write:
        #save_fig('plots/'+args.mode+'_xi_diff', draw_rectangles=True)
    #plt.pause(.1)


    '''
    convert the xi-list into a dict with the number of
    used telescopes as keys '''
    xi_vs_tel = {}
    for xi, ntel in zip(xis, NTels):
        if math.isnan(xi.value):
            continue
        if ntel not in xi_vs_tel:
            xi_vs_tel[ntel] = [xi/angle_unit]
        else:
            xi_vs_tel[ntel].append(xi/angle_unit)

    '''
    create a list of energy bin-edges and -centres for violin plots '''
    Energy_edges = np.linspace(2, 8, 13)
    Energy_centres = (Energy_edges[1:]+Energy_edges[:-1])/2.

    '''
    convert the xi-list in to an energy-binned dict with
    the bin centre as keys '''
    xi_vs_energy = {}
    for en, xi in zip(EnMC, xis):
        if math.isnan(xi.value):
            continue
        ''' get the bin number this event belongs into '''
        sbin = np.digitize(np.log10(en/energy_unit), Energy_edges)-1
        ''' the central value of the bin is the key for the dictionary '''
        if Energy_centres[sbin] not in xi_vs_energy:
            xi_vs_energy[Energy_centres[sbin]]  = [xi/angle_unit]
        else:
            xi_vs_energy[Energy_centres[sbin]] += [xi/angle_unit]

    '''
    plotting the angular error as violin plots with binning in
    number of telescopes an shower energy '''
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in xi_vs_tel.values()],
                   [a for a in xi_vs_tel.keys()],
                   points=60, widths=.75, showextrema=True, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in xi_vs_energy.values()],
                   [a for a in xi_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=True, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.grid()
    if args.write:
        save_fig('plots/_xi_vs_E_NTel_{}'.format(args.mode))

    plt.pause(.1)

    '''
    convert the diffs-list into a dict with the number of
    used telescopes as keys '''
    diff_vs_tel = {}
    for diff, ntel in zip(diffs, NTels):
        if ntel not in diff_vs_tel:
            diff_vs_tel[ntel] = [(diff/dist_unit)]
        else:
            diff_vs_tel[ntel].append((diff/dist_unit))

    '''
    convert the diffs-list in to an energy-binned dict with
    the bin centre as keys '''
    diff_vs_energy = {}
    for en, diff in zip(EnMC, diffs):
        ''' get the bin number this event belongs into '''
        sbin = np.digitize(np.log10(en/energy_unit), Energy_edges)-1
        ''' the central value of the bin is the key for the dictionary '''
        if Energy_centres[sbin] not in diff_vs_energy:
            diff_vs_energy[Energy_centres[sbin]]  = [(diff/dist_unit)]
        else:
            diff_vs_energy[Energy_centres[sbin]] += [(diff/dist_unit)]

    '''
    plotting the core position error as violin plots with binning in
    number of telescopes an shower energy '''
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in diff_vs_tel.values()],
                   [a for a in diff_vs_tel.keys()],
                   points=60, widths=.75, showextrema=True, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in diff_vs_energy.values()],
                   [a for a in diff_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=True, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()
    if args.write:
        save_fig('plots/dist_vs_E_NTel_{}'.format(args.mode))
    plt.show()

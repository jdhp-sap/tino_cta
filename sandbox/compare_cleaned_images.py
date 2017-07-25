from sys import exit, path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe"))
from datapipe.denoising.wavelets_mrfilter import WaveletTransform
wavelet_transform = WaveletTransform()


from glob import glob
import argparse

from itertools import chain

import numpy as np

from astropy import units as u

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

from extract_and_crop_simtel_images import crop_astri_image
from datapipe.denoising.wavelets_mrfilter import WaveletTransform

from ctapipe.instrument import CameraGeometry
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.io.hessio import hessio_event_source
from ctapipe.image.geometry_converter import *

# since this is in ./sandbox need to append the software path for now
path.append(expandvars("$CTA_SOFT/tino_cta"))
from modules.ImageCleaning import *
from helper_functions import *

from copy import deepcopy

global tel_geom


fig = None
from ctapipe.visualization import CameraDisplay
def transform_and_clean_hex_image(pmt_signal, cam_geom, optical_foclen, photo_electrons):
    rot_x, rot_y = unskew_hex_pixel_grid(cam_geom.pix_x, cam_geom.pix_y,
                                         cam_geom.cam_rotation)
    colors = cm.inferno(pmt_signal/max(pmt_signal))

    x_edges, y_edges, x_scale = get_orthogonal_grid_edges(rot_x, rot_y)

    new_geom, new_signal = convert_geometry_1d_to_2d(
        cam_geom, pmt_signal, cam_geom.cam_id)

    unrot_geom, unrot_signal = convert_geometry_back(
        new_geom, new_signal, cam_geom.cam_id)

    square_mask = new_geom.mask
    print(args.raw)
    cleaned_img = wavelet_transform(new_signal,
                                    raw_option_string=args.raw)

    unrot_img = cleaned_img[square_mask]
    unrot_colors = cm.inferno(unrot_img/max(unrot_img))

    cleaned_img_ik = kill_isolpix(cleaned_img, threshold=.5)
    unrot_img_ik = cleaned_img_ik[square_mask]
    unrot_colors_ik = cm.inferno(unrot_img_ik/max(unrot_img_ik))

    square_image_add_noise = np.copy(new_signal)
    square_image_add_noise[~square_mask] = \
        np.random.normal(0.13, 5.77, np.count_nonzero(~square_mask))

    square_image_add_noise_cleaned = wavelet_transform(square_image_add_noise,
                                                       raw_option_string=args.raw)

    square_image_add_noise_cleaned_ik = kill_isolpix(square_image_add_noise_cleaned,
                                                     threshold=1.5)
    unrot_geom, unrot_noised_signal = convert_geometry_back(
        new_geom, cleaned_img_ik, cam_geom.cam_id)
        # new_geom, square_image_add_noise_cleaned_ik, cam_geom.cam_id)

    global fig
    global cb1, ax1
    global cb2, ax2
    global cb3, ax3
    global cb4, ax4
    global cb5, ax5
    global cb6, ax6
    global cb7, ax7
    global cb8, ax8
    global cb9, ax9
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig.delaxes(ax1)
        fig.delaxes(ax2)
        fig.delaxes(ax3)
        fig.delaxes(ax4)
        fig.delaxes(ax5)
        fig.delaxes(ax6)
        fig.delaxes(ax7)
        fig.delaxes(ax8)
        fig.delaxes(ax9)
        cb1.remove()
        cb2.remove()
        cb3.remove()
        cb4.remove()
        cb5.remove()
        cb6.remove()
        cb7.remove()
        cb8.remove()
        cb9.remove()

    ax1 = fig.add_subplot(333)
    disp1 = CameraDisplay(cam_geom, image=photo_electrons, ax=ax1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("photo-electron image")
    disp1.cmap = plt.cm.inferno
    disp1.add_colorbar()
    cb1 = disp1.colorbar

    ax2 = fig.add_subplot(336)
    disp2 = CameraDisplay(cam_geom, image=pmt_signal, ax=ax2)
    plt.gca().set_aspect('equal', adjustable='box')
    disp2.cmap = plt.cm.inferno
    disp2.add_colorbar()
    cb2 = disp2.colorbar
    plt.title("noisy image")

    ax3 = fig.add_subplot(331)
    # plt.hist2d(rot_x, rot_y, bins=(x_edges, y_edges), cmap=cm.inferno,
    #            weights=pmt_signal)
    plt.imshow(new_signal, interpolation='none', cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("noisy, slanted image")
    cb3 = plt.colorbar()

    ax4 = fig.add_subplot(334)
    plt.imshow(cleaned_img, interpolation='none', cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, slanted image, islands not killed")
    cb4 = plt.colorbar()
    ax4.set_axis_off()

    ax5 = fig.add_subplot(337)
    plt.imshow(np.sqrt(cleaned_img_ik), interpolation='none', cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, slanted image, islands killed")
    cb5 = plt.colorbar()
    ax5.set_axis_off()



    ax6 = fig.add_subplot(332)
    plt.imshow(square_image_add_noise, interpolation='none', cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("slanted image, noise added")
    cb6 = plt.colorbar()
    ax6.set_axis_off()


    ax7 = fig.add_subplot(335)
    plt.imshow(np.sqrt(square_image_add_noise_cleaned), interpolation='none',
               cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("slanted image, noise added, cleaned")
    cb7 = plt.colorbar()
    ax7.set_axis_off()

    ax8 = fig.add_subplot(338)
    plt.imshow(square_image_add_noise_cleaned_ik, interpolation='none',
               cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("slanted image, noise added, cleaned, islands killed")
    cb8 = plt.colorbar()
    ax8.set_axis_off()


    ax9 = fig.add_subplot(339)
    disp9 = CameraDisplay(unrot_geom, image=unrot_noised_signal,
                          ax=ax9)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, original geometry, islands killed")
    disp9.cmap = plt.cm.inferno
    disp9.add_colorbar()
    cb9 = disp9.colorbar

    plt.suptitle(cam_geom.cam_id)
    plt.subplots_adjust(top=0.94, bottom=.08, left=0, right=.96, hspace=.41, wspace=.08)

    plt.pause(.1)
    response = input("press return to continue")
    if response != "":
        exit()


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")
    args = parser.parse_args()

    if args.infile_list:
        filenamelist = ["{}/{}".format(args.indir, f) for f in args.infile_list]
    elif args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
    else:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))

    # filenamelist = ["/local/home/tmichael/software/corsika_simtelarray/Data/sim_telarray/cta-ultra6/0.0deg/Data/gamma_20deg_180deg_run43___cta-prod3-demo_desert-2150m-Paranal_Tino_pureSignal_1e3GeV_PntSrc.simtel.gz"]

    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)


    cam_geom = {}

    # define here which telescopes to loop over
    allowed_tels = None
    # allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    # allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = range(34, 40)  # use the array of FlashCams instead
    for filename in sorted(filenamelist)[:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            for tel_id in event.dl0.tels_with_data:

                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])

                if cam_geom[tel_id].pix_type == "rectangular":
                    continue

                # if cam_geom[tel_id].cam_id != "DigiCam":
                #     continue

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

                if False:
                    ''' TODO switch to cleaning module if still desired '''
                else:

                    transform_and_clean_hex_image(cal_signal,
                                                  cam_geom[tel_id],
                                                  event.inst.optical_foclen[tel_id],
                                                  event.mc.tel[tel_id].photo_electron_image)
                    continue

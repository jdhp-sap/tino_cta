from sys import exit, path
from os.path import expandvars

from glob import glob
from copy import deepcopy
import argparse
import time

from itertools import chain

import numpy as np

from astropy import units as u

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

from datapipe.denoising.wavelets_mrfilter import WaveletTransform

from ctapipe.instrument import CameraGeometry

from ctapipe.io.hessio import hessio_event_source
from ctapipe.image.geometry_converter import *
from ctapipe.visualization import CameraDisplay
from ctapipe.calib import CameraCalibrator


# since this is in ./sandbox need to append the software path for now
path.append(expandvars("$CTA_SOFT/tino_cta"))
from modules.ImageCleaning import *
from helper_functions import *


from datapipe.denoising.wavelets_mrfilter import WaveletTransform
wavelet_transform = WaveletTransform()

fig = None
global tel_geom


def transform_and_clean_hex_image(pmt_signal, cam_geom, photo_electrons):

    start_time = time.time()

    colors = cm.inferno(pmt_signal/max(pmt_signal))

    new_geom, new_signal = convert_geometry_1d_to_2d(
        cam_geom, pmt_signal, cam_geom.cam_id)

    print("rot_signal", np.count_nonzero(np.isnan(new_signal)))

    square_mask = new_geom.mask
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
        new_geom, square_image_add_noise_cleaned_ik, cam_geom.cam_id)

    end_time = time.time()
    print(end_time - start_time)

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

    #
    ax6 = fig.add_subplot(332)
    plt.imshow(square_image_add_noise, interpolation='none', cmap=cm.inferno,
               origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("slanted image, noise added")
    cb6 = plt.colorbar()
    ax6.set_axis_off()

    #
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

    try:
        ax9 = fig.add_subplot(339)
        disp9 = CameraDisplay(unrot_geom, image=unrot_noised_signal,
                              ax=ax9)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("cleaned, original geometry, islands killed")
        disp9.cmap = plt.cm.inferno
        disp9.add_colorbar()
        cb9 = disp9.colorbar
    except:
        pass

    plt.suptitle(cam_geom.cam_id)
    plt.subplots_adjust(top=0.94, bottom=.08, left=0, right=.96, hspace=.41, wspace=.08)

    plt.pause(.1)
    response = input("press return to continue")
    if response != "":
        exit()


def transform_and_clean_hex_samples(pmt_samples, cam_geom):

    # rotate all samples in the image to a rectangular image
    rot_geom, rot_samples = convert_geometry_1d_to_2d(
        cam_geom, pmt_samples, cam_geom.cam_id)

    print("rot samples.shape:", rot_samples.shape)

    # rotate the samples back to hex image
    unrot_geom, unrot_samples = convert_geometry_back(rot_geom, rot_samples,
                                                      cam_geom.cam_id)

    global fig
    global cb1, ax1
    global cb2, ax2
    global cb3, ax3
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig.delaxes(ax1)
        fig.delaxes(ax2)
        fig.delaxes(ax3)
        cb1.remove()
        cb2.remove()
        cb3.remove()

    ax1 = fig.add_subplot(221)
    disp1 = CameraDisplay(rot_geom, image=np.sum(rot_samples, axis=-1), ax=ax1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("rotated image")
    disp1.cmap = plt.cm.inferno
    disp1.add_colorbar()
    cb1 = disp1.colorbar

    ax2 = fig.add_subplot(222)
    disp2 = CameraDisplay(cam_geom, image=np.sum(pmt_samples, axis=-1), ax=ax2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("original image")
    disp2.cmap = plt.cm.inferno
    disp2.add_colorbar()
    cb2 = disp2.colorbar

    ax3 = fig.add_subplot(223)
    disp3 = CameraDisplay(unrot_geom, image=np.sum(unrot_samples, axis=-1), ax=ax3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("de-rotated image")
    disp3.cmap = plt.cm.inferno
    disp3.add_colorbar()
    cb3 = disp3.colorbar

    plt.pause(.1)
    response = input("press return to continue")
    if response != "":
        exit()


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")
    parser.add_argument('--time',  action='store_true',
                        help="use images sampled in time")
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

    calib = CameraCalibrator(None, None)

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
            calib.calibrate(event)
            for tel_id in event.dl0.tels_with_data:

                camera = event.inst.subarray.tel[tel_id].camera

                if camera.pix_type == "rectangular":
                    continue

                pmt_signal = event.dl1.tel[tel_id].image
                pmt_samples = np.repeat(np.squeeze(pmt_signal)[:, None], 25, axis=1)

                print("gain, space, time")
                print("pmt_signal.shape:", pmt_signal.shape)
                print("cleaned.shape:", event.dl1.tel[tel_id].cleaned.shape)
                print("samples.shape:", pmt_samples.shape)
                print()

                if args.time:
                    transform_and_clean_hex_samples(pmt_samples, camera)
                else:
                    transform_and_clean_hex_image(np.squeeze(pmt_signal), camera,
                                                  event.mc.tel[tel_id]
                                                       .photo_electron_image)
                continue

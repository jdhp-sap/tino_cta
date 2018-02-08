import numpy as np
from astropy import units as u

import warnings

from collections import namedtuple, OrderedDict

from ctapipe.calib import CameraCalibrator

from ctapipe.image import hillas

from ctapipe.utils.linalg import rotation_matrix_2d

from tino_cta.ImageCleaning import ImageCleaner, EdgeEvent
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.coordinates.coordinate_transformations import (
            az_to_phi, alt_to_theta, transform_pixel_position)

PreparedEvent = namedtuple("PreparedEvent",
                           ["event", "hillas_dict", "n_tels",
                            "tot_signal", "max_signals",
                            "pos_fit", "dir_fit", "h_max",
                            "err_est_pos", "err_est_dir"
                            ])


def stub(event):
    return PreparedEvent(event=event, hillas_dict=None, n_tels=None,
                         tot_signal=None, max_signals=None,
                         pos_fit=None, dir_fit=None, h_max=None,
                         err_est_pos=None, err_est_dir=None)


tel_phi = {}
tel_theta = {}
tel_orientation = (tel_phi, tel_theta)

# use this in the selection of the gain channels
np_true_false = np.array([[True], [False]])


def raise_error(message):
    raise ValueError(message)


class EventPreparer():

    # for gain channel selection
    pe_thresh = {
        "ASTRICam": 14,
        "LSTCam": 100,
        "NectarCam": 190}

    def __init__(self, calib=None, cleaner=None, hillas_parameters=None,
                 shower_reco=None, event_cutflow=None, image_cutflow=None,
                 # event/image cuts:
                 allowed_cam_ids=None, min_ntel=1, min_charge=0, min_pixel=2):
        self.calib = calib or CameraCalibrator(None, None)
        self.cleaner = cleaner or ImageCleaner(mode=None)
        self.hillas_parameters = hillas_parameters or hillas.hillas_parameters
        self.shower_reco = shower_reco or \
            raise_error("need to provide a shower reconstructor....")

        # adding cutflows and cuts for events and images
        self.event_cutflow = event_cutflow or CutFlow("EventCutFlow")
        self.image_cutflow = image_cutflow or CutFlow("ImageCutFlow")

        self.event_cutflow.set_cuts(OrderedDict([
                    ("noCuts", None),
                    ("min2Tels trig", lambda x: x < min_ntel),
                    ("min2Tels reco", lambda x: x < min_ntel),
                    ("position nan", lambda x: np.isnan(x.value).any()),
                    ("direction nan", lambda x: np.isnan(x.value).any())
                ]))

        self.image_cutflow.set_cuts(OrderedDict([
                    ("noCuts", None),
                    ("min pixel", lambda s: np.count_nonzero(s) < min_pixel),
                    ("min charge", lambda x: x < min_charge),
                    ("poor moments", lambda m: m.width <= 0 or m.length <= 0)
                ]))

    @classmethod
    def pick_gain_channel(cls, pmt_signal, cam_id):
        '''the PMTs on some (most?) cameras have 2 gain channels. select one
        according to a threshold. ultimately, this will be done IN the
        camera/telescope itself but until then, do it here
        '''

        if pmt_signal.shape[0] > 1:
            pmt_signal = np.squeeze(pmt_signal)
            pick = (cls.pe_thresh[cam_id]
                    < pmt_signal).any(axis=0) != np_true_false
            pmt_signal = pmt_signal.T[pick.T]
        else:
            pmt_signal = np.squeeze(pmt_signal)
        return pmt_signal

    def prepare_event(self, source, return_stub=False):

        for event in source:

            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            # calibrate the event
            self.calib.calibrate(event)

            # telescope loop
            tot_signal = 0
            max_signals = {}
            hillas_dict = {}
            n_tels = {"tot": len(event.dl0.tels_with_data),
                      "LST": 0, "MST": 0, "SST": 0}
            for tel_id in event.dl0.tels_with_data:
                self.image_cutflow.count("noCuts")

                camera = event.inst.subarray.tel[tel_id].camera

                # can this be improved?
                if tel_id not in tel_phi:
                    tel_phi[tel_id] = az_to_phi(event.mc.tel[tel_id].azimuth_raw * u.rad)
                    tel_theta[tel_id] = \
                        alt_to_theta(event.mc.tel[tel_id].altitude_raw*u.rad)

                    # the orientation of the camera (i.e. the pixel positions) needs to
                    # be corrected
                    camera.pix_x, camera.pix_y = \
                        transform_pixel_position(camera.pix_x, camera.pix_y)

                # count the current telescope according to its size
                tel_type = event.inst.subarray.tel[tel_id].optics.tel_type
                n_tels[tel_type] += 1

                # the camera image as a 1D array
                pmt_signal = event.dl1.tel[tel_id].image

                pmt_signal = self.pick_gain_channel(pmt_signal, camera.cam_id)

                # clean the image
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pmt_signal, new_geom = \
                            self.cleaner.clean(pmt_signal.copy(), camera)

                    if self.image_cutflow.cut("min pixel", pmt_signal) or \
                       self.image_cutflow.cut("min charge", np.sum(pmt_signal)):
                        continue

                except FileNotFoundError as e:
                    print(e)
                    continue
                except EdgeEvent:
                    continue

                # could this go into `hillas_parameters` ...?
                max_signals[tel_id] = np.max(pmt_signal)

                # do the hillas reconstruction of the images
                try:
                    moments = self.hillas_parameters(new_geom, pmt_signal)

                    # import matplotlib.pyplot as plt
                    # from mpl_toolkits.mplot3d import Axes3D
                    # from ctapipe.reco.HillasReconstructor import guess_pix_direction
                    # # NOTE this is correct: +cos(psi) ; +sin(psi)
                    # p2_x = moments.cen_x + moments.length * np.cos(moments.psi)
                    # p2_y = moments.cen_y + moments.length * np.sin(moments.psi)
                    # foclen = \
                    #     event.inst.subarray.tel[tel_id].optics.effective_focal_length
                    #
                    # dir_c, dir_1, dir_2 = guess_pix_direction(
                    #     np.array([0, moments.cen_x / u.m, p2_x / u.m]) * u.m,
                    #     np.array([0, moments.cen_y / u.m, p2_y / u.m]) * u.m,
                    #     az_to_phi(event.mc.tel[tel_id].azimuth_raw * u.rad),
                    #     alt_to_theta(event.mc.tel[tel_id].altitude_raw*u.rad),
                    #     foclen)
                    #
                    # pos = event.inst.subarray.positions[tel_id]
                    # fig = plt.figure()
                    # ax = fig.gca(projection='3d')
                    # points_c = [pos + t * dir_c * u.m for t in np.linspace(0, 5000, 3)]
                    # points_1 = [pos + t * dir_1 * u.m for t in np.linspace(0, 5000, 3)]
                    # points_2 = [pos + t * dir_2 * u.m for t in np.linspace(0, 5000, 3)]
                    # ax.plot(*np.array(points_c).T, color="g", label="tel_dir")
                    # ax.plot(*np.array(points_1).T, color="r", label="centroid")
                    # ax.plot(*np.array(points_2).T, color="b", label="offset")
                    # # ax.set_aspect("equal")
                    # plt.xlim([-1000, 1000])
                    # plt.ylim([-1000, 1000])
                    # ax.set_zlim([0, 2000])
                    # # ax.invert_yaxis()
                    # plt.xlabel("x")
                    # plt.ylabel("y")
                    # plt.legend()
                    # plt.title(tel_id)

                    # # camera display
                    # from ctapipe.visualization import CameraDisplay
                    # fig = plt.figure()
                    # disp4 = CameraDisplay(new_geom, image=pmt_signal, ax=fig.gca())
                    # hw = moments
                    # plt.scatter([0], [0], color="white", marker="P")
                    # plt.scatter([hw.cen_x/u.m], [hw.cen_y/u.m], color="r")
                    # plt.scatter([(hw.cen_x+hw.length*np.cos(hw.psi))/u.m],
                    #             [(hw.cen_y+hw.length*np.sin(hw.psi))/u.m], color="b")
                    # plt.title(tel_id)

                    # plt.show()

                    # if width and/or length are zero (e.g. when there is only only one
                    # pixel or when all  pixel are exactly in one row), the
                    # parametrisation won't be very useful: skip
                    if self.image_cutflow.cut("poor moments", moments):
                        continue

                except hillas.HillasParameterizationError as e:
                    print(e)
                    print("ignoring this camera")
                    continue

                hillas_dict[tel_id] = moments
                tot_signal += moments.size

            n_tels["reco"] = len(hillas_dict)
            if self.event_cutflow.cut("min2Tels reco", n_tels["reco"]):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # telescope loop done, now do the core fit
                    self.shower_reco.get_great_circles(
                            hillas_dict, event.inst.subarray, tel_phi, tel_theta)
                    pos_fit, err_est_pos = self.shower_reco.fit_core_crosses()
                    dir_fit, err_est_dir = self.shower_reco.fit_origin_crosses()
                    h_max = self.shower_reco.fit_h_max(
                            hillas_dict, event.inst.subarray, tel_phi, tel_theta)
            except Exception as e:
                print(e)
                if return_stub:
                    yield stub(event)
                else:
                    continue

            if self.event_cutflow.cut("position nan", pos_fit) or \
               self.event_cutflow.cut("direction nan", dir_fit):
                if return_stub:
                    yield stub(event)
                else:
                    continue

            yield PreparedEvent(event=event, hillas_dict=hillas_dict, n_tels=n_tels,
                                tot_signal=tot_signal, max_signals=max_signals,
                                pos_fit=pos_fit, dir_fit=dir_fit, h_max=h_max,
                                err_est_pos=err_est_pos, err_est_dir=err_est_dir
                                )

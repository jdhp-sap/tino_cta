import numpy as np
from astropy import units as u

from ctapipe.plotting.camera import CameraGeometry

from collections import namedtuple
PreparedEvent = namedtuple("PreparedEvent",
                           ["event", "cam_geom",
                            "hillas_dict",
                            "n_lst", "n_mst", "n_sst",
                            "tot_signal", "max_signal",
                            "pos_fit", "dir_fit",
                            "err_est_pos", "err_est_dir"
                            ])

cam_geom = {}
tel_phi = {}
tel_theta = {}
tel_orientation = (tel_phi, tel_theta)

# TODO this comes now with OpticsDescription -> adapt
LST_List = ["LSTCam"]
MST_List = ["NectarCam", "FlashCam"]
SST_List = ["ASTRICam", "SCTCam", "GATE", "DigiCam", "CHEC"]

# use this in the selection of the gain channels
np_true_false = np.array([[True], [False]])


def prepare_event(source, calib=None, cleaner=None, hillas_parameters=None,
                  shower_reco=None,
                  Eventcutflow=None, Imagecutflow=None):
    for event in source:

        Eventcutflow.count("noCuts")

        if Eventcutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
            continue

        # calibrate the event
        calib.calibrate(event)

        # telescope loop
        tot_signal = 0
        max_signal = 0
        n_lst = 0
        n_mst = 0
        n_sst = 0
        hillas_dict = {}
        for tel_id in event.dl0.tels_with_data:
            Imagecutflow.count("noCuts")

            # guessing camera geometry
            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                    event.inst.pixel_pos[tel_id][0],
                                    event.inst.pixel_pos[tel_id][1],
                                    event.inst.optical_foclen[tel_id])
                tel_phi[tel_id] = 0.*u.deg
                tel_theta[tel_id] = 20.*u.deg

            # count the current telescope according to its size
            if cam_geom[tel_id].cam_id in LST_List:
                n_lst += 1
            elif cam_geom[tel_id].cam_id in MST_List:
                n_mst += 1
            elif cam_geom[tel_id].cam_id in SST_List:
                n_sst += 1
            else:
                raise ValueError(
                        "unknown camera id: {}".format(cam_geom[tel_id].cam_id) +
                        "-- please add to corresponding list")

            if cam_geom[tel_id].cam_id is not "ASTRICam":
                continue

            pmt_signal = event.dl1.tel[tel_id].image
            if pmt_signal.shape[0] > 1:
                pick = (pmt_signal > 14).any(axis=0) != np_true_false
                pmt_signal = pmt_signal.T[pick.T]
            else:
                pmt_signal = pmt_signal.ravel()
            max_signal = np.max(pmt_signal)

            # trying to clean the image
            try:
                pmt_signal, new_geom = \
                    cleaner.clean(pmt_signal.copy(), cam_geom[tel_id])

                if np.count_nonzero(pmt_signal) < 3:
                    continue

            except FileNotFoundError as e:
                print(e)
                continue
            except EdgeEventException:
                continue

            Imagecutflow.count("cleaning")

            # trying to do the hillas reconstruction of the images
            try:
                moments = hillas_parameters(new_geom.pix_x,
                                            new_geom.pix_y,
                                            pmt_signal)

                if not (moments.width > 0 and moments.length > 0):
                    continue

            except HillasParameterizationError as e:
                print(e)
                print("ignoring this camera")
                continue

            hillas_dict[tel_id] = moments
            tot_signal += moments.size
            Imagecutflow.count("Hillas")

        if Eventcutflow.cut("min2Tels reco", len(hillas_dict)):
            continue

        try:
            # telescope loop done, now do the core fit
            shower_reco.get_great_circles(hillas_dict, event.inst,
                                          tel_phi, tel_theta)
            pos_fit, err_est_pos = shower_reco.fit_core_crosses()
            dir_fit, crossings = shower_reco.fit_origin_crosses()

            # don't have a direction error estimate yet
            err_est_dir = 0
        except Exception as e:
            print(e)
            continue

        if Eventcutflow.cut("position nan", pos_fit):
            continue

        yield PreparedEvent(event, cam_geom, hillas_dict, n_lst, n_mst, n_sst,
                            tot_signal, max_signal, pos_fit, dir_fit,
                            err_est_pos, err_est_dir
                            )

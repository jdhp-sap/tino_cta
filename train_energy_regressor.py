#!/usr/bin/env python3

from collections import namedtuple

from astropy import units as u
from helper_functions import *

from sys import exit
from glob import glob

from ctapipe.reco.HillasReconstructor import \
    HillasReconstructor, TooFewTelescopesException

from ctapipe.utils import linalg

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters

from modules.CutFlow import *
from modules.ImageCleaning import *

try:
    from ctapipe.reco.energy_regressor import *
    print("using ctapipe energy_regressor")
except ImportError:
    from modules.energy_regressor import *
    print("using tino_cta energy_regressor")

from ctapipe.calib import CameraCalibrator


pckl_write = True
pckl_load = not pckl_write

cam_id_list = [
        # 'GATE',
        # 'HESSII',
        # 'NectarCam',
        # 'LSTCam',
        # 'SST-1m',
        # 'FlashCam',
        'ASTRICam',
        # 'SCTCam',
        ]

LST_List = ["LSTCam"]
MST_List = ["NectarCam", "FlashCam"]
SST_List = ["ASTRICam", "SCTCam", "GATE", "DigiCam", "CHEC"]


EnergyFeatures = namedtuple("EnergyFeatures", (
                                "impact_dist",
                                "sum_signal_evt",
                                "max_signal_cam",
                                "sum_signal_cam",
                                "N_LST",
                                "N_MST",
                                "N_SST",
                                "width",
                                "length",
                                "skewness",
                                "kurtosis",
                                "err_est_pos"))

if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--outpath', type=str,
                        default='data/classifier_pickle/regressor'
                                '_{mode}_{wave_args}_{regressor}_{cam_id}.pkl')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    args = parser.parse_args()

    filenamelist_gamma = sorted(glob("{}/gamma/run*gz".format(args.indir)))

    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    # counting events and where they might have gone missing
    Eventcutflow = CutFlow("EventCutFlow")
    Eventcutflow.set_cut("noCuts", None)
    Eventcutflow.set_cut("min2Tels trig", lambda x: x < 2)
    Eventcutflow.set_cut("min2Tels reco", lambda x: x < 2)
    Eventcutflow.set_cut("position nan", lambda x: np.isnan(x).any())

    Imagecutflow = CutFlow("ImageCutFlow")
    Imagecutflow.set_cut("noCuts", None)
    Imagecutflow.set_cut("min charge", lambda x: x < args.min_charge)

    # pass in config and self if part of a Tool
    calib = CameraCalibrator(None, None)
    Imagecutflow.set_cut("features nan", lambda x: np.isnan(x).any())

    # use this in the selection of the gain channels
    np_true_false = np.array([[True], [False]])

    # class that wraps tail cuts and wavelet cleaning
    Cleaner = ImageCleaner(mode=args.mode, wavelet_options=args.raw,
                           skip_edge_events=False)  # args.skip_edge_events)

    # simple hillas-based shower reco
    fit = HillasReconstructor()

    Features_event_list = []
    MC_Energies = []

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    allowed_tels = None  # all telescopes
    # allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    allowed_tels = np.arange(10).tolist() + np.arange(34, 41).tolist()
    for filename in filenamelist_gamma[:14][:args.last]:

        if pckl_load:
            break

        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # event loop
        for event in source:

            Eventcutflow.count("noCuts")

            mc_shower = event.mc
            mc_shower_core = np.array([mc_shower.core_x.value,
                                       mc_shower.core_y.value]) * u.m

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
                        Cleaner.clean(pmt_signal.copy(), cam_geom[tel_id])

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
                fit.get_great_circles(hillas_dict,
                                      event.inst,
                                      tel_phi, tel_theta)
                pos_fit_cr, err_est_pos = fit.fit_core_crosses()
            except Exception as e:
                print(e)
                continue

            if Eventcutflow.cut("position nan", pos_fit_cr):
                continue

            pos_fit = pos_fit_cr

            Eventcutflow.count("position fit")

            # now prepare the features for the classifier
            features_evt = {}
            NTels = len(hillas_dict)
            for tel_id in hillas_dict.keys():
                Imagecutflow.count("pre-features")

                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                moments = hillas_dict[tel_id]

                impact_dist_sim = linalg.length(tel_pos-mc_shower_core)
                impact_dist_rec = linalg.length(tel_pos-pos_fit)
                features_tel = EnergyFeatures(
                            impact_dist_rec/u.m,
                            tot_signal,
                            max_signal,
                            moments.size,
                            n_lst, n_mst, n_sst,
                            # the distance of the shower core from the centre
                            # since the camera might lose efficiency towards the edge
                            # (not a good idea for on-axis point-source MC)
                            # (moments.cen_x**2 +
                            #  moments.cen_y**2) / u.m**2,
                            # orientation of the hillas ellipsis wrt. the camera centre
                            # (moments.phi -
                            #  moments.psi)/u.deg,
                            moments.width/u.m,
                            moments.length/u.m,
                            moments.skewness,
                            moments.kurtosis,
                            err_est_pos/u.m
                          )

                if Imagecutflow.cut("features nan", features_tel):
                    continue

                cam_id = cam_geom[tel_id].cam_id
                if cam_id in features_evt:
                    features_evt[cam_id] += [features_tel]
                else:
                    features_evt[cam_id] = [features_tel]

            if len(features_evt):
                Features_event_list.append(features_evt)
                MC_Energies.append(mc_shower.energy)

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break

    print()

    if pckl_load:
        print("reading pickle")
        from sklearn.externals import joblib
        Features_event_list = \
            joblib.load("./data/{}_regression_features.pkl".format(args.mode))
        MC_Energies = joblib.load("./data/{}_regression_energy.pkl".format(args.mode))
    elif pckl_write:
        print("writing pickle")
        from sklearn.externals import joblib
        joblib.dump(Features_event_list,
                    "./data/{}_regression_features.pkl".format(args.mode))
        joblib.dump(MC_Energies, "./data/{}_regression_energy.pkl".format(args.mode))

    print("length of features:")
    print(len(Features_event_list))

    reg_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
                  'random_state': 0}

    # try neural network
    if False:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        mlp_kwargs = {'random_state': 1, 'alpha': 1e-5,
                      'hidden_layer_sizes': (150, 100, 50, 50, 25, 10)}
        mlp_reg = MLPRegressor(**mlp_kwargs)

        sskal = StandardScaler()
        reg_kwargs = {"regressor": Pipeline,
                      "steps": [("sskal", sskal), ("mlp_reg", mlp_reg)]}
    elif False:
        from sklearn.svm import SVR
        reg_kwargs = {"regressor": SVR}

    reg_kwargs['cam_id_list'] = cam_id_list
    reg = EnergyRegressor(**reg_kwargs)
    print(reg)

    reg.fit(*reg.reshuffle_event_list(Features_event_list, MC_Energies))

    # save the regressor to disk
    if args.store:
        reg.save(args.outpath.format(**{
                            "mode": args.mode,
                            "wave_args": (args.raw or "mixed").replace(' ', '')
                                                              .replace(',', ''),
                            "regressor": reg, "cam_id": "{cam_id}"}))

    if args.plot:
        # extract and show the importance of the various training features
        try:
            reg.show_importances(EnergyFeatures._fields)
            plt.tight_layout()
            plt.suptitle("{} ** {}".format(
                "wavelets" if args.mode == "wave" else "tailcuts",
                reg))
            if args.write:
                save_fig('{}/regression_importance_{}_{}_{}'.format(args.plots_dir,
                         args.mode,
                         (args.raw or "misc").replace(' ', '').replace(',', ''),
                         reg))
        except AttributeError as e:
            print("{} does not support feature importances".format(reg))
            print(e)

        # plot E_reco / E_MC ratio for a few cross-validations
        from sklearn.model_selection import KFold
        NBins = 41
        bins = np.tile(np.linspace(-2, 3, NBins), (2, 1))
        relE_bins = np.stack((np.linspace(-1, 1, NBins),
                              np.linspace(-2, 3, NBins)))

        Epred_hist = {}
        relE_Err_hist = {}
        for cam_id in ["combined"] + [k for k in reg.model_dict]:
            Epred_hist[cam_id] = np.histogram2d([], [], bins=bins)[0]
            relE_Err_hist[cam_id] = np.histogram2d([], [], bins=relE_bins)[0]

        kf = KFold(n_splits=4)
        X, y = np.array(Features_event_list), convert_astropy_array(MC_Energies)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            reg = EnergyRegressor(**reg_kwargs)
            reg.fit(*reg.reshuffle_event_list(X_train, y_train))

            y_score = reg.predict_by_event(X_test)
            y_score_dict = reg.predict_by_telescope_type(X_test)

            for evt, mce in zip(y_score_dict, y_test):
                for cam_id, pred in evt.items():
                    Epred_hist[cam_id] += np.histogram2d([np.log10(pred/u.TeV)],
                                                         [np.log10(mce/u.TeV)],
                                                         bins=bins)[0]
                    relE_Err_hist[cam_id] += np.histogram2d([(pred-mce)/mce],
                                                            [np.log10(pred/u.TeV)],
                                                            # [np.log10(mce/u.TeV)],
                                                            bins=relE_bins)[0]

            Epred_hist["combined"] += np.histogram2d(np.log10(y_score["mean"]/u.TeV),
                                                     np.log10(y_test/u.TeV),
                                                     bins=bins)[0]
            relE_Err_hist["combined"] += np.histogram2d((y_score["mean"]-y_test)/y_test,
                                                        np.log10(y_score["mean"]/u.TeV),
                                                        # np.log10(y_test/u.TeV),
                                                        bins=relE_bins)[0]

        # calculate number of rows and columns to plot camera-type plots in a grid
        n_tel_types = len(Epred_hist)
        n_cols = np.ceil(np.sqrt(n_tel_types)).astype(int)
        n_rows = np.ceil(n_tel_types / n_cols).astype(int)

        # predicted energy vs. Monte Carlo energy
        fig, axs = plt.subplots(figsize=(8, 6), nrows=n_rows, ncols=n_cols)
        plt.suptitle(" ** ".join(
                ['"migration matrices"',
                 "wavelets" if args.mode == "wave" else "tailcuts",
                 str(reg)]))

        # and for the various telescope types separately
        for i, (cam_id, thishist) in enumerate(Epred_hist.items()):
            plt.sca(axs.ravel()[i])
            plt.imshow(thishist, interpolation='none', origin='lower',
                       extent=[*bins[1, [0, -1]], *bins[0, [0, -1]]], aspect="auto")
            plt.plot(*bins[:, [0, -1]])
            plt.xlabel('log10(E_MC / TeV)')
            plt.ylabel('log10(E_reco / TeV)')
            plt.title(cam_id)
            plt.colorbar()

        plt.subplots_adjust(left=0.11, right=0.97, hspace=0.39, wspace=0.29)

        # switch off superfluous axes
        for j in range(i+1, n_rows*n_cols):
            axs.ravel()[j].axis('off')

        if args.write:
            save_fig('{}/energy_migration_{}_{}_{}'.format(args.plots_dir,
                     args.mode, (args.raw or "misc").replace(" ", ""), reg))

        #
        # 2D histogram E_rel_error vs E_mc
        fig, axs = plt.subplots(figsize=(8, 6), nrows=n_rows, ncols=n_cols)
        plt.suptitle(" ** ".join(
                ['relative Energy Error',
                 "wavelets" if args.mode == "wave" else "tailcuts",
                 str(reg)]))

        # and for the various telescope types separately
        for i, (cam_id, thishist) in enumerate(relE_Err_hist.items()):
            plt.sca(axs.ravel()[i])
            plt.imshow(thishist, interpolation='none', origin='lower',
                       extent=[*relE_bins[1, [0, -1]], *relE_bins[0, [0, -1]]],
                       aspect="auto")
            plt.plot(relE_bins[1, [0, -1]], (0, 0))
            plt.xlabel('log10(E_reco / TeV)')
            plt.ylabel('(E_reco-E_MC)/ E_MC')
            plt.title(cam_id)
            plt.colorbar()

            # get a 2D array of the cumulative sums along the "error axis"
            # (not the energy axis)
            cum_sum = np.cumsum(thishist, axis=0)
            # along the energy axis, get the index for the median, 32 and 68 percentile
            median_args = np.argmax(cum_sum > cum_sum[-1, :]*.50, axis=0)
            low_er_args = np.argmax(cum_sum > cum_sum[-1, :]*.32, axis=0)
            hih_er_args = np.argmax(cum_sum > cum_sum[-1, :]*.68, axis=0)

            bin_centres_x = (relE_bins[1, 1:]+relE_bins[1, :-1])/2
            bin_centres_y = (relE_bins[0, 1:]+relE_bins[0, :-1])/2
            medians = bin_centres_y[median_args]
            low_ers = medians - relE_bins[0, low_er_args]
            hih_ers = relE_bins[0, hih_er_args+1] - medians
            mask = median_args > 0

            # plot the median together with the error bars
            # (but only where the median index is larger than zero)
            plt.errorbar(bin_centres_x[mask], medians[mask],
                         yerr=[low_ers[mask],
                               hih_ers[mask]],
                         ls="", marker="o", ms=3, c="darkgreen")

        plt.subplots_adjust(left=0.11, right=0.97, hspace=0.39, wspace=0.29)

        # switch off superfluous axes
        for j in range(i+1, n_rows*n_cols):
            axs.ravel()[j].axis('off')

        if args.write:
            save_fig('{}/energy_relative_error_{}_{}_{}'.format(args.plots_dir,
                     args.mode, (args.raw or "misc").replace(" ", ""), reg))

        plt.show()

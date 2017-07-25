#!/usr/bin/env python3

from helper_functions import *
from astropy import units as u
from collections import namedtuple

from sys import exit, path
from os.path import expandvars
from glob import glob
from itertools import chain

from ctapipe.utils import linalg

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters

from modules.CutFlow import *
from modules.ImageCleaning import *

try:
    from ctapipe.reco.event_classifier import *
    print("using ctapipe event_classifier")
except ImportError:
    from modules.event_classifier import *
    print("using tino_cta event_classifier")

try:
    from ctapipe.reco.FitGammaHillas import \
        FitGammaHillas as HillasReconstructor, TooFewTelescopesException
except ImportError:
    from ctapipe.reco.HillasReconstructor import \
        HillasReconstructor, TooFewTelescopesException

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


ClassifierFeatures = namedtuple("ClassifierFeatures", (
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
                        default='data/classifier_pickle/classifier'
                                '_{mode}_{wave_args}_{classifier}_{cam_id}.pkl')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    args = parser.parse_args()

    filenamelist_gamma = sorted(glob("{}/gamma/run*gz".format(args.indir)))
    filenamelist_proton = sorted(glob("{}/proton/run*gz".format(args.indir)))

    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    # counting events and where they might have gone missing
    Eventcutflow = {"p": CutFlow("EventCutFlow Protons"),
                    "g": CutFlow("EventCutFlow Gammas")}
    Imagecutflow = {"p": CutFlow("ImageCutFlow Protons"),
                    "g": CutFlow("ImageCutFlow Gammas")}
    for E in Eventcutflow.values():
        E.set_cut("noCuts", None)
        E.set_cut("min2Tels trig", lambda x: x < 2)
        E.set_cut("min2Tels reco", lambda x: x < 2)

    for I in Imagecutflow.values():
        I.set_cut("position nan", lambda x: np.isnan(x).any())
        I.set_cut("features nan", lambda x: np.isnan(x).any())

    # pass in config and self if part of a Tool
    calib = CameraCalibrator(None, None)

    # use this in the selection of the gain channels
    np_true_false = np.array([[True], [False]])

    # class that wraps tail cuts and wavelet cleaning
    Cleaner = ImageCleaner(mode=args.mode, wavelet_options=args.raw,
                           skip_edge_events=False)  # args.skip_edge_events)

    # simple hillas-based shower reco
    fit = HillasReconstructor()

    Features_event_list = {"g": [], "p": []}
    MC_Energies = {"g": [], "p": []}

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    allowed_tels = None  # all telescopes
    # allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    allowed_tels = np.arange(10).tolist() + np.arange(34, 41).tolist()
    for filenamelist_class in [filenamelist_gamma[:14],
                               filenamelist_proton[:100]]:

        if pckl_load:
            break

        signal_handler.stop = False

        # get type of event for the classifier
        # assume that there are only gamma and proton as event class
        # if `filenamelist_gamma` is empty, though `cl` will be set to proton, the
        # `filename` loop will be empty, so no mislabelling will occur
        cl = "g" if "gamma" in " ".join(filenamelist_class) else "p"

        for filename in filenamelist_class[:args.last]:
            print("filename = {}".format(filename))

            source = hessio_event_source(filename,
                                         allowed_tels=allowed_tels,
                                         max_events=args.max_events)

            # event loop
            for event in source:

                Eventcutflow[cl].count("noCuts")

                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                if Eventcutflow[cl].cut("min2Tels trig", len(event.dl0.tels_with_data)):
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
                    Imagecutflow[cl].count("noCuts")

                    pmt_signal = event.dl1.tel[tel_id].image
                    if pmt_signal.shape[0] > 1:
                        pick = (pmt_signal > 14).any(axis=0) != np_true_false
                        pmt_signal = pmt_signal.T[pick.T]
                    else:
                        pmt_signal = pmt_signal.ravel()

                    max_signal = np.max(pmt_signal)

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

                    Imagecutflow[cl].count("cleaning")

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

                    Imagecutflow[cl].count("Hillas")

                if Eventcutflow[cl].cut("min2Tels reco", len(hillas_dict)):
                    continue

                try:
                    # telescope loop done, now do the core fit
                    fit.get_great_circles(hillas_dict,
                                          event.inst,
                                          tel_phi, tel_theta)
                    pos_fit, err_est_pos = fit.fit_core_crosses()
                except Exception as e:
                    print(e)
                    continue

                if Imagecutflow[cl].cut("position nan", pos_fit):
                    continue

                Eventcutflow[cl].count("position fit")

                # now prepare the features for the classifier
                features_evt = {}
                for tel_id in hillas_dict.keys():
                    Imagecutflow[cl].count("pre-features")

                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                    moments = hillas_dict[tel_id]

                    impact_dist_rec = linalg.length(tel_pos-pos_fit)
                    features_tel = ClassifierFeatures(
                                impact_dist_rec/u.m,
                                tot_signal,
                                max_signal,
                                moments.size,
                                n_lst,
                                n_mst,
                                n_sst,
                                moments.width/u.m,
                                moments.length/u.m,
                                moments.skewness,
                                moments.kurtosis,
                                err_est_pos/u.m)

                    if Imagecutflow[cl].cut("features nan", features_tel):
                        continue

                    cam_id = cam_geom[tel_id].cam_id
                    if cam_id in features_evt:
                        features_evt[cam_id] += [features_tel]
                    else:
                        features_evt[cam_id] = [features_tel]

                if len(features_evt):
                    Features_event_list[cl].append(features_evt)
                    MC_Energies[cl].append(mc_shower.energy)

                if signal_handler.stop:
                    break
            if signal_handler.stop:
                break

    print()

    if pckl_load:
        print("reading pickle")
        from sklearn.externals import joblib
        Features_event_list = \
            joblib.load("./data/{}_classification_features.pkl".format(args.mode))
    elif pckl_write:
        print("writing pickle")
        from sklearn.externals import joblib
        joblib.dump(Features_event_list,
                    "./data/{}_classification_features.pkl".format(args.mode))

    print("length of features:")
    for cl, feat in Features_event_list.items():
        print(cl, len(feat))

    trainFeatures = []
    trainClasses = []

    # reduce the number of events by random draw so that they are the same
    # in gammas and protons; then add them to a single list
    minEvents = min([len(Features) for Features in Features_event_list.values()])
    for cl in Features_event_list.keys():
        trainFeatures = np.append(
                            trainFeatures,
                            np.random.choice(Features_event_list[cl], minEvents, False))
        trainClasses += [cl]*minEvents

    # use default random forest classifier
    clf_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
                  'random_state': 0}

    # try neural network
    if False:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        mlp_kwargs = {'random_state': 1, 'alpha': 1e-5,
                      'hidden_layer_sizes': (150, 150, 150)}
        mlp_clf = MLPClassifier(**mlp_kwargs)
        sskal = StandardScaler()

        clf_kwargs = {"classifier": Pipeline,
                      "steps": [("sskal", sskal), ("mlp_clf", mlp_clf)]}

        # hidden layer trials
        # (100, 50)    AUC: 92-93
        # (100, 100)   AUC: 93,92,92,93,92
        # (100, 100)   AUC: 92,93,92,93,93

    clf_kwargs['cam_id_list'] = cam_id_list

    clf = EventClassifier(**clf_kwargs)
    print(clf)

    train_features, train_classes = clf.reshuffle_event_list(trainFeatures, trainClasses)

    # save the classifier to disk
    if args.store:
        clf.fit(train_features, train_classes)
        clf.save(args.outpath.format(**{
                            "mode": args.mode,
                            "wave_args": "mixed",
                            # args.raw.replace(' ', '').replace(',', ''),
                            "classifier": clf, "cam_id": "{cam_id}"}))

    if args.plot:
        # extract and show the importance of the various training features
        try:
            clf.fit(train_features, train_classes)
            clf.show_importances(ClassifierFeatures._fields)
            plt.suptitle("{} ** {}".format(
                "wavelets" if args.mode == "wave" else "tailcuts",
                clf))
            if args.write:
                save_fig('{}/classification_importance_{}_{}_{}'.format(args.plots_dir,
                         args.mode, args.raw.replace(" ", ""), clf))
        except AttributeError as e:
            print(e)
            print("{} does not support feature importances".format(clf))

        # plot area under curve for a few cross-validations
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
        colours = ["darkorange", "r", "b", "g", "black"]

        plt.figure()
        lw = 2
        # plot a diagonal line that represents purely random choices
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.suptitle("{} ** {}".format(
            "wavelets" if args.mode == "wave" else "tailcuts", clf))

        X, y = np.array(trainFeatures), np.array(trainClasses)
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = EventClassifier(**clf_kwargs)

            X_train_flat, y_train_flat = clf.reshuffle_event_list(X_train, y_train)
            clf.fit(X_train_flat, y_train_flat)

            y_score = clf.predict_proba_by_event(X_test)[:, 0]

            fpr, tpr, _ = roc_curve(y_test == "g", y_score)
            roc_auc = auc(fpr, tpr)
            print("area under curve:", roc_auc)

            plt.plot(fpr, tpr, color=colours[i],
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.pause(.1)

        # legend does not update dynamically, needs to be called at the end
        plt.legend(loc="lower right", title="cross validation")

        if args.write:
            save_fig('{}/classification_receiver_operating_curve_{}_{}_{}'
                     .format(args.plots_dir, args.mode,
                             (args.raw or "misc").replace(" ", ""), clf))
        plt.pause(.1)

        # plot gammaness as function of number of telescopes
        histos = {'g': None, 'p': None}
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.19, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = EventClassifier(**clf_kwargs)

            clf.fit(*clf.reshuffle_event_list(X_train, y_train))

            for cl in histos.keys():
                test = X_test[y_test == cl]
                NTels = []
                for evt in test:
                    n_tels = 0
                    for c, t in evt.items():
                        n_tels += len(t)
                    NTels.append(n_tels)

                try:
                    y_score = clf.predict_proba_by_event(test)[:, 0]
                except:
                    continue

                temp = np.histogram2d(NTels, y_score,
                                      bins=(range(2, 10), np.linspace(0, 1, 11)))[0].T

                if histos[cl] is None:
                    histos[cl] = temp
                else:
                    histos[cl] += temp

        fig = plt.figure()
        for cl in histos.keys():
            histo = histos[cl]

            ax = plt.subplot(121 if cl == "g" else 122)
            histo_normed = histo / histo.max(axis=0)
            im = ax.imshow(histo_normed, interpolation='none', origin='lower',
                           aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
            cb = fig.colorbar(im, ax=ax)
            ax.set_xlabel("NTels")
            ax.set_ylabel("drifted gammaness")
            plt.title("protons" if cl == 'p' else "gamma")

        plt.suptitle("{} ** {}".format("wavelets" if args.mode == "wave" else "tailcuts",
                                       clf))
        plt.subplots_adjust(left=0.10, right=0.95, wspace=0.33)

        if args.write:
            save_fig('{}/classification_gammaness_vs_NTel_{}_{}_{}'
                     .format(args.plots_dir, args.mode,
                             (args.raw or "misc").replace(" ", ""), clf))

        plt.show()

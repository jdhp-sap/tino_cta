#!/usr/bin/env python3

import os
from os.path import expandvars

from collections import namedtuple
from glob import glob

# PyTables
import tables as tb

from ctapipe.io.hessio import hessio_event_source

from ctapipe.utils import linalg
from ctapipe.utils.CutFlow import CutFlow

from ctapipe.reco.event_classifier import EventClassifier

from ctapipe.reco.HillasReconstructor import \
    HillasReconstructor, TooFewTelescopes

from modules.prepare_event import EventPreparer
from modules.ImageCleaning import ImageCleaner

from helper_functions import *

# your favourite units here
energy_unit = u.TeV
angle_unit = u.deg
dist_unit = u.m

# for which cam_id to generate a models
cam_id_list = [
        # 'GATE',
        # 'HESSII',
        'NectarCam',
        'LSTCam',
        'DigiCam',
        # 'SST-1m',
        # 'FlashCam',
        # 'ASTRICam',
        # 'SCTCam',
        ]


ClassifierFeatures = namedtuple(
    "ClassifierFeatures", (
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
        "h_max",
        "err_est_pos",
        "err_est_dir"
    ))


parser = make_argparser()
parser.add_argument('-o', '--outpath', type=str,
                    default='data/classifier_pickle/classifier'
                            '_{mode}_{cam_id}_{classifier}.pkl')
parser.add_argument('--check', action='store_true',
                    help="run a self check on the classification")
parser.add_argument('--unify', action='store_true',
                    help="weight the images to 1 per (channel and camera)")
args = parser.parse_args()


feature_file_gammas = tb.open_file(f"data/features_{args.mode}_gamma.h5", mode="r")
feature_file_proton = tb.open_file(f"data/features_{args.mode}_proton.h5", mode="r")

feature_file = feature_file_gammas
features = {"LSTCam": [[row[name] for name in ClassifierFeatures._fields] for row in
                       feature_file.root.feature_events_lst],
            "DigiCam": [[row[name] for name in ClassifierFeatures._fields] for row in
                        feature_file.root.feature_events_dig],
            "NectarCam": [[row[name] for name in ClassifierFeatures._fields] for row in
                          feature_file.root.feature_events_nec]}

energies = {"LSTCam": [row["MC_Energy"] for row in
                       feature_file.root.feature_events_lst],
            "DigiCam": [row["MC_Energy"] for row in
                        feature_file.root.feature_events_dig],
            "NectarCam": [row["MC_Energy"] for row in
                          feature_file.root.feature_events_nec]}

classes = {}
for cam_id, feats in features.items():
    classes[cam_id] = ["g"] * len(feats)

#
# now protons
feature_file = feature_file_proton
features["LSTCam"] += [[row[name] for name in ClassifierFeatures._fields] for row in
                       feature_file.root.feature_events_lst]
features["DigiCam"] += [[row[name] for name in ClassifierFeatures._fields] for row in
                        feature_file.root.feature_events_dig]
features["NectarCam"] += [[row[name] for name in ClassifierFeatures._fields] for row in
                          feature_file.root.feature_events_nec]
energies["LSTCam"] += [row["MC_Energy"] for row in
                       feature_file.root.feature_events_lst]
energies["DigiCam"] += [row["MC_Energy"] for row in
                        feature_file.root.feature_events_dig]
energies["NectarCam"] += [row["MC_Energy"] for row in
                          feature_file.root.feature_events_nec]
for cam_id, feats in features.items():
    classes[cam_id] += ["p"] * (len(feats) - len(classes[cam_id]))
    energies[cam_id] = np.array(energies[cam_id]) * energy_unit


telescope_weights = {}
for cam_id, cl in classes.items():
    print(cam_id)
    cl = np.array(cl)
    telescope_weights[cam_id] = np.ones_like(cl, dtype=np.float)
    if args.unify:
        telescope_weights[cam_id][cl == 'g'] = \
            1 / np.count_nonzero(cl == 'g')
        telescope_weights[cam_id][cl == 'p'] = \
            1 / np.count_nonzero(cl == 'p')

    print("number of g:", np.count_nonzero(cl == 'g'))
    print("number of p:", np.count_nonzero(cl == 'p'))
    print()

# use default random forest classifier
clf_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
              'random_state': 0, 'cam_id_list': cam_id_list}
classifier = EventClassifier(**clf_kwargs)
classifier.fit(features, classes, telescope_weights)

if args.store:
    classifier.save(args.outpath.format(mode=args.mode,
                                        classifier=classifier,
                                        cam_id="{cam_id}"))

fig = classifier.show_importances(ClassifierFeatures._fields)
fig.set_size_inches(15, 10)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)
    for label in ax.get_xmajorticklabels():
        label.set_horizontalalignment("right")
plt.subplots_adjust(top=0.9, bottom=0.135, left=0.034, right=0.98,
                    hspace=0.478, wspace=0.08)
plt.pause(.1)

# do some cross-validation now
if args.check:
    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # takes care of image cleaning
    cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           skip_edge_events=False, island_cleaning=True)

    # the class that does the shower reconstruction
    shower_reco = HillasReconstructor()

    preper = EventPreparer(
        cleaner=cleaner, shower_reco=shower_reco,
        event_cutflow=Eventcutflow, image_cutflow=Imagecutflow,
        # event/image cuts:
        allowed_cam_ids=[],  # [] or None means: all
        min_ntel=2,
        min_charge=args.min_charge, min_pixel=3)
    Imagecutflow.add_cut("features nan", lambda x: np.isnan(x).any())

    energy_mc = []
    gammaness = []
    true_class = []

    filenamelist_gamma = sorted(glob("{}/gamma/*gz".format(args.indir)))[50:]
    filenamelist_gamma = sorted(glob(expandvars("$CTA_DATA/Prod3b/Paranal/*simtel.gz")))
    filenamelist_proton = \
        sorted(glob(expandvars("$CTA_DATA/Prod3b/Paranal/proton/*simtel.gz")))[50:]

    allowed_tels = prod3b_tel_ids("L+N+D")
    for filenamelist in [filenamelist_gamma, filenamelist_proton]:
        channel = "g" if "gamma" in "".join(filenamelist) else "p"
        for i, filename in enumerate(filenamelist[:5][:args.last]):

            print(f"{i} -- filename = {filename}")

            source = hessio_event_source(filename,
                                         allowed_tels=allowed_tels,
                                         #  max_events=args.max_events)
                                         max_events=400)

            # loop that cleans and parametrises the images and performs the reconstruction
            for (event, hillas_dict, n_tels,
                 tot_signal, max_signals, pos_fit, dir_fit, h_max,
                 err_est_pos, err_est_dir) in preper.prepare_event(source):

                # now prepare the features for the classifier
                cls_features_evt = {}
                for tel_id in hillas_dict.keys():
                    Imagecutflow.count("pre-features")

                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                    moments = hillas_dict[tel_id]

                    impact_dist = linalg.length(tel_pos - pos_fit)

                    cls_features_tel = ClassifierFeatures(
                        impact_dist=impact_dist / u.m,
                        sum_signal_evt=tot_signal,
                        max_signal_cam=max_signals[tel_id],
                        sum_signal_cam=moments.size,
                        N_LST=n_tels["LST"],
                        N_MST=n_tels["MST"],
                        N_SST=n_tels["SST"],
                        width=moments.width / u.m,
                        length=moments.length / u.m,
                        skewness=moments.skewness,
                        kurtosis=moments.kurtosis,
                        h_max=h_max / u.m,
                        err_est_pos=err_est_pos / u.m,
                        err_est_dir=err_est_dir / u.deg
                    )

                    if np.isnan(cls_features_tel).any():
                        continue

                    # any nans reconstructed?
                    Imagecutflow.count("features nan")

                    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id

                    try:
                        cls_features_evt[cam_id] += [cls_features_tel]
                    except KeyError:
                        cls_features_evt[cam_id] = [cls_features_tel]

                if not cls_features_evt:
                    continue

                predict_proba = classifier.predict_proba_by_event([cls_features_evt])
                gammaness.append(predict_proba[0, 0])
                true_class.append(channel)
                energy_mc.append(event.mc.energy / energy_unit)

    gammaness = np.array(gammaness)
    true_class = np.array(true_class)
    energy_mc = np.array(energy_mc)

    # plot area under curve for a few cross-validations
    from sklearn.metrics import roc_curve, auc

    plt.figure()
    lw = 2
    # plot a diagonal line that represents purely random choices
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    # plt.suptitle("{} ** {}".format(
    #     "wavelets" if args.mode == "wave" else "tailcuts", clf))

    fpr, tpr, _ = roc_curve(true_class == "g", gammaness)
    roc_auc = auc(fpr, tpr)
    print("area under curve:", roc_auc)

    plt.plot(fpr, tpr,
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend()
    plt.pause(.1)

plt.show()

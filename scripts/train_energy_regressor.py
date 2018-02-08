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

from ctapipe.reco.energy_regressor import *

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


EnergyFeatures = namedtuple(
    "EnergyFeatures", (
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
                    default='data/classifier_pickle/regressor'
                            '_{mode}_{cam_id}_{regressor}.pkl')
parser.add_argument('--check', action='store_true',
                    help="run a self check on the classification")
args = parser.parse_args()


feature_file_gammas = tb.open_file(f"data/features_{args.mode}_gamma.h5", mode="r")

feature_file = feature_file_gammas
features = {"LSTCam": [[row[name] for name in EnergyFeatures._fields] for row in
                       feature_file.root.feature_events_lst],
            "DigiCam": [[row[name] for name in EnergyFeatures._fields] for row in
                        feature_file.root.feature_events_dig],
            "NectarCam": [[row[name] for name in EnergyFeatures._fields] for row in
                          feature_file.root.feature_events_nec]}

energies = {"LSTCam": np.array([row["MC_Energy"] for row in
                                feature_file.root.feature_events_lst]) * energy_unit,
            "DigiCam": np.array([row["MC_Energy"] for row in
                                 feature_file.root.feature_events_dig]) * energy_unit,
            "NectarCam": np.array([row["MC_Energy"] for row in
                                   feature_file.root.feature_events_nec]) * energy_unit}

# use default random forest regressor
reg_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
              'random_state': 0, 'cam_id_list': cam_id_list}
regressor = EnergyRegressor(**reg_kwargs)
regressor.fit(features, energies)

if args.store:
    regressor.save(args.outpath.format(mode=args.mode,
                                       regressor=regressor,
                                       cam_id="{cam_id}"))

fig = regressor.show_importances(EnergyFeatures._fields)
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
    energy_rec = []


    filenamelist_gamma = sorted(glob("{}/gamma/*gz".format(args.indir)))
    filenamelist_gamma = sorted(glob(expandvars("$CTA_DATA/Prod3b/Paranal/*simtel.gz")))

    allowed_tels = prod3b_tel_ids("L+N+D")
    for filename in filenamelist_gamma[:5][:args.last]:

        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # loop that cleans and parametrises the images and performs the reconstruction
        for (event, hillas_dict, n_tels,
             tot_signal, max_signals, pos_fit, dir_fit, h_max,
             err_est_pos, err_est_dir) in preper.prepare_event(source):

            # now prepare the features for the classifier
            cls_features_evt = {}
            reg_features_evt = {}
            for tel_id in hillas_dict.keys():
                Imagecutflow.count("pre-features")

                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                moments = hillas_dict[tel_id]

                impact_dist = linalg.length(tel_pos - pos_fit)

                reg_features_tel = EnergyFeatures(
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

                if np.isnan(reg_features_tel).any():
                    continue

                # any nans reconstructed?
                Imagecutflow.count("features nan")

                cam_id = event.inst.subarray.tel[tel_id].camera.cam_id

                try:
                    reg_features_evt[cam_id] += [reg_features_tel]
                except KeyError:
                    reg_features_evt[cam_id] = [reg_features_tel]

            if not reg_features_evt:
                continue

            predict_energ = regressor.predict_by_event([reg_features_evt])["mean"][0]
            energy_rec.append(predict_energ / energy_unit)
            energy_mc.append(event.mc.energy / energy_unit)

    energy_mc = np.array(energy_mc)
    energy_rec = np.array(energy_rec)

    e_bin_edges = np.logspace(-2, np.log10(330), 20) * u.TeV
    e_bin_centres = (e_bin_edges[:-1] + e_bin_edges[1:]) / 2
    e_bin_fine_edges = np.logspace(-2, 2.5, 100) * u.TeV
    e_bin_fine_centres = (e_bin_fine_edges[:-1] + e_bin_fine_edges[1:]) / 2


    def percentiles(values, bin_values, bin_edges, percentile):
        percentiles_binned = \
            np.squeeze(np.full((len(bin_edges) - 1, len(values.shape)), np.inf))
        for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            try:
                percentiles_binned[i] = \
                    np.percentile(values[(bin_values > bin_l) &
                                         (bin_values < bin_h)], percentile)
            except IndexError:
                pass
        return percentiles_binned.T


    # (reco Energy - MC Energy) / reco Energy vs. reco Energy 2D histograms
    fig, ax = plt.subplots(1, 1)
    counts, _, _ = np.histogram2d(
                energy_rec, (energy_rec - energy_mc) / energy_rec,
                bins=(e_bin_fine_edges, np.linspace(-2, 1.5, 50)))
    ax.pcolormesh(e_bin_fine_edges.value, np.linspace(-2, 1.5, 50),
                  np.sqrt(counts.T))
    plt.plot(e_bin_fine_edges.value[[0, -1]], [0, 0],
             color="darkgreen")
    ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC}) / E_\mathrm{reco}$")
    ax.set_xscale("log")
    plt.grid()
    plt.subplots_adjust(left=.1, wspace=.1)

    # energy resolution
    rel_DeltaE_w = np.abs(energy_rec - energy_mc) / energy_rec
    DeltaE68_w_ebinned = percentiles(rel_DeltaE_w, energy_rec,
                                     e_bin_edges.value, 68)
    plt.figure()
    plt.plot(e_bin_centres.value, DeltaE68_w_ebinned, label="gamma -- wave",
             marker='^', color="darkred")
    plt.title("Energy Resolution")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{reco}$")
    plt.gca().set_xscale("log")
    plt.grid()
    plt.legend()

    plt.show()

    # (reco Energy - MC Energy) vs. MC Energy 2D histograms
    fig, ax = plt.subplots(1, 1)
    counts, _, _ = np.histogram2d(
                energy_mc,
                (energy_rec - energy_mc) / energy_mc,
                bins=(e_bin_fine_edges, np.linspace(-1, 2, 50)))
    ax.pcolormesh(e_bin_fine_edges.value, np.linspace(-1, 2, 50),
                  np.sqrt(counts.T))
    plt.plot(e_bin_fine_edges.value[[0, -1]], [0, 0],
             color="darkgreen")
    ax.set_xlabel(r"$E_\mathrm{MC}$ / TeV")
    ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC}) / E_\mathrm{MC}$")
    ax.set_xscale("log")
    plt.grid()
    plt.subplots_adjust(left=.1, wspace=.1)


    rel_DeltaE_w = np.abs(energy_rec - energy_mc) / energy_mc
    DeltaE68_w_ebinned = percentiles(rel_DeltaE_w, energy_mc,
                                     e_bin_edges.value, 68)
    plt.figure()
    plt.plot(e_bin_centres.value, DeltaE68_w_ebinned, label="gamma -- wave",
             marker='^', color="darkred")
    plt.title("Energy Resolution")
    plt.xlabel(r"$E_\mathrm{mc}$ / TeV")
    plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{mc}$")
    plt.gca().set_xscale("log")
    plt.grid()
    plt.legend()


plt.show()

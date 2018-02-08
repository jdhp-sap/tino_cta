#!/usr/bin/env python

import sys
from os.path import expandvars
import argparse

import numpy as np

from scipy import optimize, interpolate

from astropy import units as u
from astropy.table import Table

import pandas as pd

from matplotlib import pyplot as plt

import irf_builder as irf


def correct_off_angle(data, origin=None):
    import ctapipe.utils.linalg as linalg
    origin = origin or linalg.set_phi_theta(90 * u.deg, 20 * u.deg)

    reco_dirs = linalg.set_phi_theta(data["phi"] * u.deg.to(u.rad),
                                     data["theta"] * u.deg.to(u.rad)).T
    off_angles = np.arccos(np.clip(np.dot(reco_dirs, origin), -1., 1.)) * u.rad
    data["off_angle"] = off_angles.to(u.deg)


def diff_to_X_purity(gammaness, events, target, signal=None):
    signal = signal or ['g']

    n_sig = 0.
    n_bck = 0.
    for ch, ev in events.items():
        if ch in signal:
            n_sig += np.count_nonzero(ev[ev["gammaness"] > gammaness[0]])
        else:
            n_bck += np.count_nonzero(ev[ev["gammaness"] > gammaness[0]])
    if n_sig + n_bck == 0:
        return 1
    else:
        return ((n_sig / (n_sig + n_bck)) - target)**2


def get_gammaness_for_target_purity(events, target):

    gammaness = []

    print("doing optimisation now")
    for e_low, e_high, e_mid in zip(irf.e_bin_edges[:-1],
                                    irf.e_bin_edges[1:],
                                    irf.e_bin_centres):
        # print(f"e_low = {e_low}")
        cut_events = {}
        for ch, ev in events.items():
            cut_events[ch] = ev[(ev[irf.mc_energy_name] > e_low) &
                                (ev[irf.mc_energy_name] < e_high)]
        res = optimize.differential_evolution(
            diff_to_X_purity,
            maxiter=1000, popsize=10,
            args=(cut_events, target),
            bounds=[(0, 1)],
        )
        if res.success:
            gammaness.append(res.x[0])
        else:
            gammaness.append(0)
    return gammaness


parser = argparse.ArgumentParser(description='')
parser.add_argument('--indir',
                    default=expandvars("$CTA_SOFT/tino_cta/data/prod3b/paranal_LND"))
parser.add_argument('--infile', type=str, default="classified_events")
parser.add_argument('--r_scale', type=float, default=5.)
parser.add_argument('--purity', type=float, default=.9)
parser.add_argument('-k', type=int, default=1, help="order of spline interpolation")

args = parser.parse_args()

modes = ["wave", "tail"]

irf.r_scale = args.r_scale
irf.alpha = irf.r_scale**-2


# reading the reconstructed and classified events
all_events = {}
for mode in modes:
    all_events[mode] = {}
    for c, channel in irf.plotting.channel_map.items():
        all_events[mode][c] = \
            pd.read_hdf(f"{args.indir}/{args.infile}_{channel}_{mode}.h5")

# FUCK FUCK FUCK FUCK
for c in irf.plotting.channel_map:
    correct_off_angle(all_events["wave"][c])


# # # # # #
# determine optimal bin-by-bin cut values and fit splines to them
def main(purity, all_events, args):
    xi_cuts = {}
    ga_cuts = {}
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for mode in modes:
        xi_cuts[mode] = irf.irfs.get_angular_resolution(all_events[mode])
        if True:
            ga_cuts[mode] = get_gammaness_for_target_purity(all_events[mode], args.purity)
        else:
            ga_cuts["tail"] = [0.00010797506398901868, 0.1755767099797223,
                               0.75512375889607319, 0.95315723793511065,
                               0.9918075064217492, 0.9950950534358125,
                               0.99035880010037181, 0.97337397150095684,
                               0.95496618056823357, 0.95326166540164403,
                               0.94959951856254943, 0.93741002949880992,
                               0.92214253489467879, 0.90634486820581173,
                               0.89279990617402805, 0.87853035376596433,
                               0.87288220808693284, 0.878079494796003,
                               0.8557482520923928]
            ga_cuts["wave"] = [0.00032108517271961512, 0.058295752528910283,
                               0.74258793260276645, 0.95654804819785022,
                               0.98909886863845242, 0.99337538669131609,
                               0.98612760716648884, 0.96106703954423656,
                               0.9352384171690159, 0.94319842943480203,
                               0.93910583837329253, 0.93673481528761127,
                               0.91640867834111783, 0.90671758177465511,
                               0.89253479608630659, 0.87534228528795499,
                               0.86440677324889814, 0.87561903218727266,
                               0.83751966865987681]

    #     plt.sca(axes[0])
    #     plt.plot(irf.e_bin_centres, xi_cuts[mode]['g'], label=mode)
    #     plt.gca().set_xscale("log")
    #     plt.gca().set_yscale("log")
    #
    #     plt.sca(axes[1])
    #     plt.plot(irf.e_bin_centres, ga_cuts[mode], label=mode)
    #     plt.gca().set_xscale("log")
    #     plt.gca().set_yscale("log")
    #
    # plt.grid()
    # plt.tight_layout()
    # plt.legend()
    # plt.pause(.1)

    spline_xi = {}
    spline_ga = {}
    for mode in modes:
        spline_ga[mode] = interpolate.splrep(irf.e_bin_centres.value,
                                             ga_cuts[mode], k=args.k)
        spline_xi[mode] = interpolate.splrep(irf.e_bin_centres.value,
                                             xi_cuts[mode]['g'], k=args.k)

    # evaluating cuts and add columns with flags
    for mode, events in all_events.items():
        for key in events:
            events[key]["pass_gammaness"] = \
                events[key]["gammaness"] > interpolate.splev(events[key]["reco_Energy"],
                                                             spline_ga[mode])
            events[key]["pass_theta"] = \
                events[key]["off_angle"] < (1 if key == 'g' else irf.r_scale) * \
                interpolate.splev(events[key]["reco_Energy"], spline_xi[mode])

    # applying the cuts
    cut_events = dict(
        (m, irf.event_selection.apply_cuts(e, ["pass_gammaness", "pass_theta"]))
        for m, e in all_events.items())

    significance = {}
    for mode, ev in cut_events.items():
        n_g = len(ev['g'])
        n_p = len(ev['p'])
        n_e = len(ev['e'])

        significance[mode] = n_g / (n_g + n_p + n_e)**.5
        print(f"mode: {mode} -- significance: {significance[mode]}")

    return significance


if __name__ == "__main__":
    purities = np.linspace(.9, .99, 20)
    sig_w = []
    sig_t = []
    for pur in purities:
        print(f"purity {pur}")
        sig = main(all_events=all_events, purity=pur, args=args)
        sig_w.append(sig["wave"])
        sig_t.append(sig["tail"])
        print()

    plt.figure()
    plt.plot(purities, sig_w, label="wave")
    plt.plot(purities, sig_t, label="tail")
    plt.xlabel("purity")
    plt.ylabel(r"$N_g / \sqrt{N_g+N_p+N_e}$")
    plt.legend()
    plt.show()

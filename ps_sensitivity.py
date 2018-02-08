#!/usr/bin/env python3
import os
import glob
import numpy as np

# pandas data frames
import pandas as pd

from astropy import units as u

from itertools import chain

from scipy import optimize
from scipy import interpolate

from helper_functions import *

from ctapipe.analysis.sensitivity import (SensitivityPointSource, e_minus_2,
                                          crab_source_rate, cr_background_rate)

from os.path import expandvars
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)


# define edges to sort events in
e_bin_edges = np.logspace(-2, np.log10(330), 20) * u.TeV
e_bin_centres = (e_bin_edges[:-1] + e_bin_edges[1:]) / 2
e_bin_fine_edges = np.logspace(-2, 2.5, 100) * u.TeV
e_bin_fine_centres = (e_bin_fine_edges[:-1] + e_bin_fine_edges[1:]) / 2

# MC energy ranges:
# gammas: 0.003 to 330 TeV
# proton: 0.004 to 600 TeV
edges_gammas = np.logspace(np.log10(0.003), np.log10(330), 28) * u.TeV
edges_proton = np.logspace(np.log10(0.004), np.log10(600), 30) * u.TeV
edges_electr = np.logspace(np.log10(0.003), np.log10(330), 28) * u.TeV
sensitivity_energy_bin_edges = np.logspace(-2, 2.5, 24) * u.TeV
# sensitivity_energy_bin_edges = np.logspace(-2, 2.5, 17)*u.TeV
edges_gammas = sensitivity_energy_bin_edges
edges_proton = sensitivity_energy_bin_edges
edges_electr = sensitivity_energy_bin_edges

# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg * u.cm**2 * u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

# scale MC events to this reference time
observation_time = 50 * u.h

# scaling factors for the on and off regions
r_scale = 5
alpha = r_scale**-2

channel_map = {'g': "gamma", 'p': "proton", 'e': "electron"}
channel_color_map = {'g': "orange", 'p': "blue", 'e': "red"}
channel_marker_map = {'g': 's', 'p': '^', 'e': 'v'}
channel_linestyle_map = {'g': '-', 'p': '--', 'e': ':'}


def electron_spectrum(e_true_tev):
    """Cosmic-Ray Electron spectrum CTA version, with Fermi Shoulder, in
    units of :math:`\mathrm{TeV^{-1} s^{-1} m^{-2} sr^{-1}}`

    .. math::
       {dN \over dE dA dt d\Omega} =

    """
    e_true_tev /= u.TeV
    number = (6.85e-5 * e_true_tev**-3.21 +
              3.18e-3 / (e_true_tev * 0.776 * np.sqrt(2 * np.pi)) *
              np.exp(-0.5 * (np.log(e_true_tev / 0.107) / 0.776)**2))
    return number * u.Unit("TeV**-1 s**-1 m**-2 sr**-1")


def powerlaw(energy, index, norm, norm_energy=1.0):
    return norm * (energy / norm_energy)**(-index)


def exponential_cutoff(energy, cutoff_energy):
    return np.exp(-energy / cutoff_energy)


def hess_crab_spectrum(e_true_tev, fraction=1.0):
    e_true_tev /= u.TeV
    norm = fraction * u.Quantity(3.76e-11, "cm**-2 s**-1 TeV**-1")
    return powerlaw(e_true_tev, norm=norm,
                    index=2.39, norm_energy=1.0) \
        * exponential_cutoff(e_true_tev, cutoff_energy=14.3)
# crab_source_rate = hess_crab_spectrum


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


def correct_off_angle(data, origin=None):
    import ctapipe.utils.linalg as linalg
    origin = origin or linalg.set_phi_theta(90 * u.deg, 20 * u.deg)

    reco_dirs = linalg.set_phi_theta(data["phi"] * u.deg.to(u.rad),
                                     data["theta"] * u.deg.to(u.rad)).T
    off_angles = np.arccos(np.clip(np.dot(reco_dirs, origin), -1., 1.)) * u.rad
    data["off_angle"] = off_angles.to(u.deg)


def calculate_sensitivities(events, energy_bin_edges, alpha, n_draws=1):
    SensCalc = SensitivityPointSource(
        reco_energies={'g': events['g']['MC_Energy'].values * u.TeV,
                       'p': events['p']['MC_Energy'].values * u.TeV,
                       'e': events['e']['MC_Energy'].values * u.TeV},
        mc_energies={'g': events['g']['MC_Energy'].values * u.TeV,
                     'p': events['p']['MC_Energy'].values * u.TeV,
                     'e': events['e']['MC_Energy'].values * u.TeV},
        flux_unit=flux_unit)

    SensCalc.generate_event_weights(
        n_simulated_events={'g': meta_gammas["n_simulated"],
                            'p': meta_proton["n_simulated"],
                            'e': meta_electr["n_simulated"]},
        generator_areas={'g': np.pi * (meta_gammas["gen_radius"] * u.m)**2,
                         'p': np.pi * (meta_proton["gen_radius"] * u.m)**2,
                         'e': np.pi * (meta_electr["gen_radius"] * u.m)**2},
        observation_time=observation_time,
        spectra={'g': crab_source_rate,
                 'p': cr_background_rate,
                 'e': electron_spectrum},
        e_min_max={'g': (meta_gammas["e_min"], meta_gammas["e_max"]) * u.TeV,
                   'p': (meta_proton["e_min"], meta_proton["e_max"]) * u.TeV,
                   'e': (meta_electr["e_min"], meta_electr["e_max"]) * u.TeV},
        extensions={'p': meta_proton["diff_cone"] * u.deg,
                    'e': meta_electr["diff_cone"] * u.deg},
        generator_gamma={'g': meta_gammas["gen_gamma"],
                         'p': meta_proton["gen_gamma"],
                         'e': meta_electr["gen_gamma"]})

    SensCalc.get_sensitivity(
        alpha=alpha, n_draws=n_draws, max_background_ratio=.05,
        sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    return SensCalc


def cut_and_sensitivity(cuts, events, energy_bin_edges, alpha):
    """ throw this into a minimiser """
    ga_cut = cuts[0]
    xi_cut = cuts[1]

    cut_events = {}
    for key in events:
        cut_events[key] = events[key][
            (events[key]["gammaness"] > ga_cut) &
            # the background regions are larger to gather more statistics
            (events[key]["off_angle"] < xi_cut / (1 if key == 'g' else alpha))]

    if len(events['g']) < 10 or \
            len(events['g']) < (len(events['p']) + len(events['e'])) * 0.05:
        return 1

    SensCalc = calculate_sensitivities(
        cut_events, energy_bin_edges, n_draws=1, alpha=alpha)

    if len(SensCalc.sensitivities):
        return SensCalc.sensitivities["Sensitivity"][0]
    else:
        return 1


def get_optimal_splines(events, optimise_bin_edges, k=1):

    cut_events = {}
    cut_energies, ga_cuts, xi_cuts = [], [], []
    for elow, ehigh, emid in zip(optimise_bin_edges[:-1],
                                 optimise_bin_edges[1:],
                                 np.sqrt(optimise_bin_edges[:-1] *
                                         optimise_bin_edges[1:])):

        for key in events:
            cut_events[key] = events[key][
                (events[key]["MC_Energy"] > elow) &
                (events[key]["MC_Energy"] < ehigh)]

        res = optimize.differential_evolution(
            cut_and_sensitivity,
            bounds=[(.5, 1), (0, 0.5)],
            maxiter=1000, popsize=10,
            args=(cut_events,
                  np.array([elow / energy_unit,
                            ehigh / energy_unit]) * energy_unit,
                  alpha)
        )

        if res.success:
            cut_energies.append(emid.value)
            ga_cuts.append(res.x[0])
            xi_cuts.append(res.x[1])

    spline_ga = interpolate.splrep(cut_energies, ga_cuts, k=k)
    spline_xi = interpolate.splrep(cut_energies, xi_cuts, k=k)

    return (spline_ga, ga_cuts), (spline_xi, xi_cuts)


# ########  ##        #######  ########  ######
# ##     ## ##       ##     ##    ##    ##    ##
# ##     ## ##       ##     ##    ##    ##
# ########  ##       ##     ##    ##     ######
# ##        ##       ##     ##    ##          ##
# ##        ##       ##     ##    ##    ##    ##
# ##        ########  #######     ##     ######

def make_sensitivity_plots(SensCalc, sensitivities,
                           SensCalc_t, sensitivities_t):

    bin_centres_g = (edges_gammas[1:] + edges_gammas[:-1]) / 2.
    bin_centres_p = (edges_proton[1:] + edges_proton[:-1]) / 2.

    bin_widths_g = np.diff(edges_gammas.value)
    bin_widths_p = np.diff(edges_proton.value)

    # the point-source sensitivity binned in energy

    plt.figure()

    # draw the crab flux as a reference
    crab_bins = np.logspace(-2, 2.5, 17)
    plt.loglog(crab_bins,
               (crab_source_rate(crab_bins * u.TeV)
                * (crab_bins * u.TeV)**2).to(sensitivity_unit).value,
               color="red", ls="dashed", label="Crab Nebula")
    plt.loglog(crab_bins,
               (crab_source_rate(crab_bins * u.TeV)
                * (crab_bins * u.TeV)**2).to(sensitivity_unit).value / 10,
               color="red", ls="dashed", alpha=.66, label="Crab Nebula / 10")
    plt.loglog(crab_bins,
               (crab_source_rate(crab_bins * u.TeV)
                * (crab_bins * u.TeV)**2).to(sensitivity_unit).value / 100,
               color="red", ls="dashed", alpha=.33, label="Crab Nebula / 100")

    # some semi-official line to compare
    ref_loge, ref_sens = *(np.array([
            (-1.8, 6.87978e-11), (-1.6, 1.87765e-11),
            (-1.4, 7.00645e-12), (-1.2, 1.77677e-12), (-1.0, 8.19263e-13),
            (-0.8, 4.84879e-13), (-0.6, 3.00256e-13), (-0.4, 2.07787e-13),
            (-0.2, 1.4176e-13), (0.0, 1.06069e-13), (0.2, 8.58209e-14),
            (0.4, 6.94294e-14), (0.6, 6.69301e-14), (0.8, 7.61169e-14),
            (1.0, 7.13895e-14), (1.2, 9.49376e-14), (1.4, 1.25208e-13),
            (1.6, 1.91209e-13), (1.8, 3.11611e-13), (2.0, 4.80354e-13)]).T),
    plt.loglog(10**ref_loge,
               ((ref_sens) * (u.erg * u.cm**2 * u.s)**(-1)).to(flux_unit).value,
               marker="s", color="black", ms=3, linewidth=1,
               label="reference")

    sens_low, sens_up = (
        (sensitivities["Sensitivity"] -
         sensitivities["Sensitivity_low"]).to(flux_unit) *
        sensitivities["Energy"].to(u.erg)**2,
        (sensitivities["Sensitivity_up"] -
         sensitivities["Sensitivity"]).to(flux_unit) *
        sensitivities["Energy"].to(u.erg)**2)

    # plt.errorbar(
    plt.loglog(
        sensitivities["Energy"] / energy_unit,
        (sensitivities["Sensitivity"].to(flux_unit) *
         sensitivities["Energy"].to(u.erg)**2).to(sensitivity_unit).value,
        # (sens_low.value, sens_up.value),
        color="darkred",
        marker="s",
        label="wavelets")

    # plt.loglog(
    #     sensitivities["Energy"] / energy_unit,
    #     (sensitivities["Sensitivity_base"].to(flux_unit) *
    #      sensitivities["Energy"].to(u.erg)**2).to(sensitivity_unit).value,
    #     color="darkgreen",
    #     marker="^",
    #     ls="",
    #     label="wavelets (no upscale)")

    # tailcuts
    sens_low_t, sens_up_t = (
        (sensitivities_t["Sensitivity"] -
         sensitivities_t["Sensitivity_low"]).to(flux_unit) *
        sensitivities_t["Energy"].to(u.erg)**2,
        (sensitivities_t["Sensitivity_up"] -
         sensitivities_t["Sensitivity"]).to(flux_unit) *
        sensitivities_t["Energy"].to(u.erg)**2)

    # plt.errorbar(
    plt.loglog(
        sensitivities_t["Energy"] / energy_unit,
        (sensitivities_t["Sensitivity"].to(flux_unit) *
         sensitivities_t["Energy"].to(u.erg)**2).to(sensitivity_unit).value,
        # (sens_low_t.value, sens_up_t.value),
        color="darkorange",
        marker="s", ls="--",
        label="tailcuts")

    # plt.loglog(
    #     sensitivities_t["Energy"].to(energy_unit),
    #     (sensitivities_t["Sensitivity_base"].to(flux_unit) *
    #      sensitivities_t["Energy"].to(u.erg)**2),
    #     color="darkblue",
    #     marker="v",
    #     ls="",
    #     label="tailcuts (no upscale)")

    plt.legend(title="Obsetvation Time: {}".format(observation_time), loc=1)
    plt.xlabel(r'$E_\mathrm{reco}$' + ' / {:latex}'.format(energy_unit))
    plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
    plt.gca().set_xscale("log")
    plt.grid()
    plt.xlim([1e-2, 2e2])
    plt.ylim([5e-15, 5e-10])
    if args.write:
        save_fig(args.plots_dir + "sensitivity")

    # plot the sensitivity ratios
    try:
        plt.figure()
        plt.semilogx(sensitivities_t["Energy"].to(energy_unit)[1:-2].value,
                     (sensitivities_t["Sensitivity"].to(flux_unit) *
                      sensitivities_t["Energy"].to(u.erg)**2)[1:-2] /
                     (sensitivities["Sensitivity"].to(flux_unit) *
                      sensitivities["Energy"].to(u.erg)**2)[1:-2],
                     label=r"Sens$_\mathrm{tail}$ / Sens$_\mathrm{wave}$"
                     )
        plt.legend()
        # plt.semilogx(sensitivities_t["Energy"].to(energy_unit)[[0, -1]],
        #              [1, 1], ls="--", color="gray")
        # plt.xlim(sensitivities_t["Energy"].to(energy_unit)[[0, -1]].value)
        # plt.ylim([.25, 1.1])
        plt.xlabel('E / {:latex}'.format(energy_unit))
        plt.ylabel("ratio")
        if args.write:
            save_fig(args.plots_dir + "sensitivity_ratio")
    except:
        plt.close()

    # plot a sky image of the events
    # useless since too few MC background events left
    if False:
        fig2 = plt.figure()
        plt.hexbin(
            [(ph - 180) * np.sin(th * u.deg) for
                ph, th in zip(chain(gammas['phi'], proton['phi']),
                              chain(gammas['theta'], proton['theta']))],
            [a for a in chain(gammas['theta'], proton['theta'])],
            gridsize=41, extent=[-2, 2, 18, 22],
            C=[a for a in chain(weights['g'], weights['p'])],
            bins='log'
        )
        plt.colorbar().set_label("log(Number of Events)")
        plt.axes().set_aspect('equal')
        plt.xlabel(r"$\sin(\vartheta) \cdot (\varphi-180) / ${:latex}"
                   .format(angle_unit))
        plt.ylabel(r"$\vartheta$ / {:latex}".format(angle_unit))
        if args.write:
            save_fig("plots/skymap")


def show_gammaness(events, suptitle=None):

    gammas = events['g']
    proton = events['p']
    electr = events['e']

    # 1D gammaness
    e_slices = np.array([0.03, 0.1, 1, np.inf]) * u.TeV
    for e_step in sliding_window(e_slices, step_size=1, window_size=2):
        plt.figure()
        for key in events:
            plt.hist(events[key]["gammaness"][(events[key]["MC_Energy"] > e_step[0]) &
                                              (events[key]["MC_Energy"] < e_step[1])],
                     bins=100, normed=True, alpha=.5,
                     ls=channel_linestyle_map[key],
                     label="--".join([channel_map[key], suptitle]),
                     color=channel_color_map[key])
        plt.suptitle(" to ".join([str(e_step[0]), str(e_step[1])]))
        plt.xlabel("gammaness")
        plt.ylabel("normalised Events")
        plt.legend(title="integral normalised")

        if args.write:
            save_fig(args.plots_dir +
                     "_".join(["gammaness", suptitle, str(e_step[0]), str(e_step[1])]))
        plt.pause(.1)

    gamm_bins = np.linspace(0, 1, 101)
    NTels_bins = np.linspace(0, 50, 21)[:]
    energy_bins = np.linspace(-2, 2.5, 16)[1:]
    energy_bin_centres = (energy_bins[:-1] + energy_bins[1:]) / 2

    # gammaness vs. number of telescopes

    gamm_vs_ntel_g = np.histogram2d(gammas["NTels_reco"], gammas["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_p = np.histogram2d(proton["NTels_reco"], proton["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_e = np.histogram2d(electr["NTels_reco"], electr["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_g = gamm_vs_ntel_g / gamm_vs_ntel_g.sum(axis=0)
    gamm_vs_ntel_p = gamm_vs_ntel_p / gamm_vs_ntel_p.sum(axis=0)
    gamm_vs_ntel_e = gamm_vs_ntel_e / gamm_vs_ntel_e.sum(axis=0)

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(131)
    im = ax1.imshow(np.sqrt(gamm_vs_ntel_g), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.,
                    cmap=plt.cm.inferno)
    ax1.set_xlabel("NTels")
    ax1.set_ylabel("gammaness")
    ax1.set_title("gammas")

    ax2 = plt.subplot(132, sharey=ax1)
    im = ax2.imshow(np.sqrt(gamm_vs_ntel_p), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.)
    ax2.set_xlabel("NTels")
    ax2.set_title("protons")

    ax3 = plt.subplot(133, sharey=ax1)
    im = ax3.imshow(np.sqrt(gamm_vs_ntel_e), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.)
    ax3.set_xlabel("NTels")
    ax3.set_title("electrons")

    cb = fig.colorbar(im, ax=[ax1, ax2, ax3], label="sqrt(event fraction per column)")

    if suptitle:
        plt.suptitle(suptitle)

    if args.write:
        save_fig(args.plots_dir + "gammaness_vs_n_tel_" + suptitle)

    plt.pause(.1)

    #
    # gammaness vs. reconstructed energy

    gamm_vs_e_reco_g = np.histogram2d(
            np.log10(gammas["reco_Energy"]), gammas["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_p = np.histogram2d(
            np.log10(proton["reco_Energy"]), proton["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_e = np.histogram2d(
            np.log10(electr["reco_Energy"]), electr["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_g = gamm_vs_e_reco_g / gamm_vs_e_reco_g.sum(axis=0)
    gamm_vs_e_reco_p = gamm_vs_e_reco_p / gamm_vs_e_reco_p.sum(axis=0)
    gamm_vs_e_reco_e = gamm_vs_e_reco_e / gamm_vs_e_reco_e.sum(axis=0)

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(131)
    im = ax1.imshow(np.sqrt(gamm_vs_e_reco_g), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.,
                    cmap=plt.cm.inferno)
    ax1.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax1.set_ylabel("gammaness")
    ax1.set_title("gammas")

    ax2 = plt.subplot(132, sharey=ax1)
    im = ax2.imshow(np.sqrt(gamm_vs_e_reco_p), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.)
    ax2.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax2.set_title("protons")

    ax3 = plt.subplot(133, sharey=ax1)
    im = ax3.imshow(np.sqrt(gamm_vs_e_reco_e), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.)
    ax3.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax3.set_title("electrons")

    cb = fig.colorbar(im, ax=[ax1, ax2, ax3], label="sqrt(event fraction per E-column)")

    if suptitle:
        plt.suptitle(suptitle)

    if args.write:
        save_fig(args.plots_dir + "gammaness_vs_e_reco_" + suptitle)

    # # same as a wireframe plot
    #
    # from mpl_toolkits.mplot3d import axes3d
    # xv, yv = np.meshgrid(energy_bins, gamm_bins)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe((xv[1:, 1:]+xv[:-1, :-1])/2,
    #                   (yv[1:, 1:]+yv[:-1, :-1])/2, gamm_vs_e_reco_g,
    #                   color="darkred", label="gamma")
    # ax.plot_wireframe((xv[1:, 1:]+xv[:-1, :-1])/2,
    #                   (yv[1:, 1:]+yv[:-1, :-1])/2, gamm_vs_e_reco_p,
    #                   color="darkorange", label="proton")
    # ax.set_xlabel(r"$\log_{10}(E_\mathrm{reco})$ / TeV")
    # ax.set_ylabel("gammaness")
    # ax.set_zlabel("event fraction per E-row")
    # plt.legend()


def make_performance_plots(events_w, events_t, which=None):

    if (not which or "multiplicity" in which) and False:
        fig = plt.figure()
        plt.hist(events_w['g']["NTels_reco"], alpha=.5,
                 bins=np.arange(0, 51, 2))
        plt.hist(events_t['g']["NTels_reco"], alpha=.5,
                 bins=np.arange(0, 51, 2))
        if args.write:
            save_fig(args.plots_dir + "multiplicity")
        plt.pause(.1)

    if not which or "multiplicity_by_size" in which:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        plt.sca(axs[0])
        plt.hist(events_w['g']["NTels_reco_lst"], alpha=.5,
                 bins=np.arange(0, 5, 1))
        plt.hist(events_t['g']["NTels_reco_lst"], alpha=.5,
                 bins=np.arange(0, 5, 1))
        plt.suptitle("LST")

        plt.sca(axs[1])
        plt.hist(events_w['g']["NTels_reco_mst"], alpha=.5,
                 bins=np.arange(0, 15, 1))
        plt.hist(events_t['g']["NTels_reco_mst"], alpha=.5,
                 bins=np.arange(0, 15, 1))
        plt.suptitle("MST")

        plt.sca(axs[2])
        plt.hist(events_w['g']["NTels_reco_sst"], alpha=.5,
                 bins=np.arange(0, 15, 1))
        plt.hist(events_t['g']["NTels_reco_sst"], alpha=.5,
                 bins=np.arange(0, 15, 1))
        plt.suptitle("SST")

        if args.write:
            save_fig(args.plots_dir + "multiplicity_by_size")
        plt.pause(.1)

    if (not which or "ang_res_verbose" in which) and False:
        fig, axes = plt.subplots(1, 2)
        n_tel_max = 50  # np.max(gammas_w["NTels_reco"])
        # plt.subplots_adjust(left=0.11, right=0.97, hspace=0.39, wspace=0.29)
        plot_hex_and_violin(events_w['g']["NTels_reco"],
                            np.log10(events_w['g']["off_angle"]),
                            np.arange(0, n_tel_max + 1, 5),
                            xlabel=r"$N_\mathrm{Tels}$",
                            ylabel=r"$\log_{10}(\xi / ^\circ)$",
                            do_hex=False, axis=axes[0],
                            extent=[0, n_tel_max, -3, 0])
        plot_hex_and_violin(np.log10(events_w['g']["reco_Energy"]),
                            np.log10(events_w['g']["off_angle"]),
                            np.linspace(-1, 3, 17),
                            xlabel=r"$\log_{10}(E_\mathrm{reco}$ / TeV)",
                            ylabel=r"$\log_{10}(\xi / ^\circ)$",
                            v_padding=0.015, axis=axes[1], extent=[-.5, 2.5, -3, 0])
        plt.suptitle("wavelet")

        if args.write:
            save_fig(args.plots_dir + "ang_res_verbose_wave")
        plt.pause(.1)

        fig, axes = plt.subplots(1, 2)
        n_tel_max = 50  # np.max(gammas_w["NTels_reco"])
        # plt.subplots_adjust(left=0.11, right=0.97, hspace=0.39, wspace=0.29)
        plot_hex_and_violin(events_t['g']["NTels_reco"],
                            np.log10(events_t['g']["off_angle"]),
                            np.arange(0, n_tel_max + 1, 5),
                            xlabel=r"$N_\mathrm{Tels}$",
                            ylabel=r"$\log_{10}(\xi / ^\circ)$",
                            do_hex=False, axis=axes[0],
                            extent=[0, n_tel_max, -3, 0])
        plot_hex_and_violin(np.log10(events_t['g']["reco_Energy"]),
                            np.log10(events_t['g']["off_angle"]),
                            np.linspace(-1, 3, 17),
                            xlabel=r"$\log_{10}(E_\mathrm{reco}$ / TeV)",
                            ylabel=r"$\log_{10}(\xi / ^\circ)$",
                            v_padding=0.015, axis=axes[1], extent=[-.5, 2.5, -3, 0])
        plt.suptitle("tailcuts")

        if args.write:
            save_fig(args.plots_dir + "ang_res_verbose_tail")
        plt.pause(.1)

    # angular resolutions

    if not which or ("ang_res" in which or "xi" in which):
        plt.figure()
        for key in events_w:
            xi_68_w = percentiles(events_w[key]["off_angle"],
                                  events_w[key]["reco_Energy"],
                                  e_bin_edges.value, 68)
            xi_68_t = percentiles(events_t[key]["off_angle"],
                                  events_t[key]["reco_Energy"],
                                  e_bin_edges.value, 68)

            plt.plot(e_bin_centres.value, xi_68_t,
                     color="darkorange",
                     marker=channel_marker_map[key],
                     ls=channel_linestyle_map[key],
                     label="--".join([channel_map[key], "tail"]))
            plt.plot(e_bin_centres.value, xi_68_w,
                     color="darkred",
                     marker=channel_marker_map[key],
                     ls=channel_linestyle_map[key],
                     label="--".join([channel_map[key], "wave"]))
        plt.title("angular resolution")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel(r"$\xi_\mathrm{68} / ^\circ$")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.grid()
        plt.legend()

        if args.write:
            save_fig(args.plots_dir + "xi")
        plt.pause(.1)

    if not which or "gammaness" in which:
        # gammaness plots
        show_gammaness(events_w, "wavelets")
        show_gammaness(events_t, "tailcuts")

    if not which or "energy_migration" in which:
        # MC Energy vs. reco Energy 2D histograms
        for events, mode in zip([events_w, events_t],
                                ["wavelets", "tailcuts"]):
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)
                counts, _, _ = np.histogram2d(events[key]["reco_Energy"],
                                              events[key]["MC_Energy"],
                                              bins=(e_bin_fine_edges, e_bin_fine_edges))
                ax.pcolormesh(e_bin_fine_edges.value, e_bin_fine_edges.value, counts.T)
                plt.plot(e_bin_fine_edges.value[[0, -1]], e_bin_fine_edges.value[[0, -1]],
                         color="darkgreen")
                plt.title(channel_map[key])
                ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
                if i == 0:
                    ax.set_ylabel(r"$E_\mathrm{MC}$ / TeV")
                ax.set_xscale("log")
                ax.set_yscale("log")
                plt.grid()

            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)

            if args.write:
                save_fig(args.plots_dir + "_".join(["energy_migration", mode]))
            plt.pause(.1)

    if not which or "DeltaE" in which:
        # (reco Energy - MC Energy) 2D histograms
        for events, mode in zip([events_w, events_t],
                                ["wavelets", "tailcuts"]):

            # (reco Energy - MC Energy) vs. reco Energy 2D histograms
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)

                counts, _, _ = np.histogram2d(
                            events[key]["reco_Energy"],
                            events[key]["reco_Energy"] - events[key]["MC_Energy"],
                            bins=(e_bin_fine_edges, np.linspace(-1, 1, 100)))
                ax.pcolormesh(e_bin_fine_edges.value, np.linspace(-1, 1, 100),
                              np.sqrt(counts.T))
                plt.plot(e_bin_fine_edges.value[[0, -1]], [0, 0],
                         color="darkgreen")
                plt.title(channel_map[key])
                ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
                if i == 0:
                    ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC})$ / TeV")
                ax.set_xscale("log")
                plt.grid()

            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)
            if args.write:
                save_fig(args.plots_dir + "_".join(["DeltaE_vs_recoE", mode]))
            plt.pause(.1)

            # (reco Energy - MC Energy) vs. MC Energy 2D histograms
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)

                counts, _, _ = np.histogram2d(
                            events[key]["MC_Energy"],
                            events[key]["reco_Energy"] - events[key]["MC_Energy"],
                            bins=(e_bin_fine_edges, np.linspace(-1, 1, 100)))
                ax.pcolormesh(e_bin_fine_edges.value, np.linspace(-1, 1, 100),
                              np.sqrt(counts.T))
                plt.plot(e_bin_fine_edges.value[[0, -1]], [0, 0],
                         color="darkgreen")
                plt.title(channel_map[key])
                ax.set_xlabel(r"$E_\mathrm{MC}$ / TeV")
                if i == 0:
                    ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC})$ / TeV")
                ax.set_xscale("log")
                plt.grid()

            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)
            if args.write:
                save_fig(args.plots_dir + "_".join(["DeltaE_vs_MCE", mode]))
            plt.pause(.1)

    if not which or "Energy_resolution" in which:
        # energy resolution as 68th percentile of the relative reconstructed error binned
        # in reconstructed energy
        rel_DeltaE_w = np.abs(events_w['g']["reco_Energy"] -
                              events_w['g']["MC_Energy"])/events_w['g']["reco_Energy"]
        DeltaE68_w_ebinned = percentiles(rel_DeltaE_w, events_w['g']["reco_Energy"],
                                         e_bin_edges.value, 68)

        rel_DeltaE_t = np.abs(events_t['g']["reco_Energy"] -
                              events_t['g']["MC_Energy"])/events_t['g']["reco_Energy"]
        DeltaE68_t_ebinned = percentiles(rel_DeltaE_t, events_t['g']["reco_Energy"],
                                         e_bin_edges.value, 68)

        plt.figure()
        plt.plot(e_bin_centres.value, DeltaE68_t_ebinned, label="gamma -- tail",
                 marker='v', color="darkorange")
        plt.plot(e_bin_centres.value, DeltaE68_w_ebinned, label="gamma -- wave",
                 marker='^', color="darkred")
        plt.title("Energy Resolution")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{reco}$")
        plt.gca().set_xscale("log")
        plt.grid()
        plt.legend()

        if args.write:
            save_fig(args.plots_dir + "Energy_resolution_vs_recoE")
        plt.pause(.1)

        # energy resolution binned in MC Energy
        for key in ['g', 'e']:
            plt.figure()
            for events, mode in zip([events_w, events_t],
                                    ["wavelets", "tailcuts"]):

                rel_DeltaE = np.abs(events[key]["reco_Energy"] -
                                    events[key]["MC_Energy"]) / events[key]["MC_Energy"]
                DeltaE68_ebinned = percentiles(rel_DeltaE, events[key]["MC_Energy"],
                                               e_bin_edges.value, 68)

                plt.plot(e_bin_centres.value, DeltaE68_ebinned,
                         label=" -- ".join([channel_map[key], mode]),
                         marker=channel_marker_map[key],
                         color="darkred" if "wave" in mode else "darkorange")
            plt.title("Energy Resolution")
            plt.xlabel(r"$E_\mathrm{MC}$ / TeV")
            plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{MC}$")
            plt.gca().set_xscale("log")
            plt.grid()
            plt.legend()

            if args.write:
                save_fig(args.plots_dir + "Energy_resolution_"
                         + channel_map[key])
            plt.pause(.1)

    if not which or "Energy_bias" in which:
        # Ebias as median of 1-E_reco/E_MC
        for key in ['g', 'e']:
            plt.figure()
            for events, mode in zip([events_w, events_t],
                                    ["wavelets", "tailcuts"]):
                Ebias = 1 - (events[key]["reco_Energy"] / events[key]["MC_Energy"])
                Ebias_medians = percentiles(Ebias, events[key]["reco_Energy"],
                                            e_bin_edges.value, 50)
                plt.plot(e_bin_centres.value, Ebias_medians,
                         label=" -- ".join([channel_map[key], mode]),
                         marker=channel_marker_map[key],
                         color="darkred" if "wave" in mode else "darkorange")
            plt.title("Energy Bias")
            plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
            plt.ylabel(r"$(1 - E_\mathrm{reco}/E_\mathrm{MC})_{50}$")
            plt.ylim([-0.2, .3])
            plt.gca().set_xscale("log")
            plt.legend()
            plt.grid()

            if args.write:
                save_fig(args.plots_dir + "Energy_bias_"
                         + channel_map[key])
            plt.pause(.1)

    if not which or any(what in which for what in ["gen_spectrum", "expected_events",
                                                   "effective_areas", "event_rate"]):
        bin_centres, bin_widths = {}, {}
        bin_centres['g'] = (edges_gammas[:-1] + edges_gammas[1:]) / 2
        bin_centres['p'] = (edges_proton[:-1] + edges_proton[1:]) / 2
        bin_centres['e'] = (edges_electr[:-1] + edges_electr[1:]) / 2
        bin_widths['g'] = np.diff(edges_gammas)
        bin_widths['p'] = np.diff(edges_proton)
        bin_widths['e'] = np.diff(edges_electr)

        # for events, mode in zip([events_t, events_w], ["tailcuts", "wavelets"]):
        for events, mode in zip([events_w], ["wavelets"]):
            SensCalc = SensitivityPointSource(
                reco_energies={'g': events['g']['reco_Energy'].values * u.TeV,
                               'p': events['p']['reco_Energy'].values * u.TeV,
                               'e': events['e']['reco_Energy'].values * u.TeV},
                mc_energies={'g': events['g']['MC_Energy'].values * u.TeV,
                             'p': events['p']['MC_Energy'].values * u.TeV,
                             'e': events['e']['MC_Energy'].values * u.TeV},
                energy_bin_edges={'g': edges_gammas,
                                  'p': edges_proton,
                                  'e': edges_electr},
                flux_unit=flux_unit)

            SensCalc.generate_event_weights(
                n_simulated_events={'g': meta_gammas["n_simulated"],
                                    'p': meta_proton["n_simulated"],
                                    'e': meta_electr["n_simulated"]},
                generator_areas={'g': np.pi *
                                 (meta_gammas["gen_radius"] * u.m)**2,
                                 'p': np.pi *
                                 (meta_proton["gen_radius"] * u.m)**2,
                                 'e': np.pi *
                                 (meta_electr["gen_radius"] * u.m)**2},
                observation_time=observation_time,
                spectra={'g': crab_source_rate,
                         'p': cr_background_rate,
                         'e': electron_spectrum},
                e_min_max={'g': (meta_gammas["e_min"],
                                 meta_gammas["e_max"]) * u.TeV,
                           'p': (meta_proton["e_min"],
                                 meta_proton["e_max"]) * u.TeV,
                           'e': (meta_electr["e_min"],
                                 meta_electr["e_max"]) * u.TeV},
                extensions={'p': meta_proton["diff_cone"] * u.deg,
                            'e': meta_electr["diff_cone"] * u.deg},
                generator_gamma={'g': meta_gammas["gen_gamma"],
                                 'p': meta_proton["gen_gamma"],
                                 'e': meta_electr["gen_gamma"]})
            SensCalc.get_effective_areas(
                generator_areas={'g': np.pi *
                                 (meta_gammas["gen_radius"] * u.m)**2,
                                 'p': np.pi *
                                 (meta_proton["gen_radius"] * u.m)**2,
                                 'e': np.pi *
                                 (meta_electr["gen_radius"] * u.m)**2},
                n_simulated_events={'g': meta_gammas["n_simulated"],
                                    'p': meta_proton["n_simulated"],
                                    'e': meta_electr["n_simulated"]},
                generator_spectra={'g': e_minus_2,
                                   'p': e_minus_2,
                                   'e': e_minus_2}
            )

            print("expected events:")
            for ch in ['g', 'p', 'e']:
                print(f"{ch}:", np.sum(SensCalc.event_weights[ch]))

            SensCalc.get_expected_events()
            SensCalc.get_expected_events_in_reco_e()

            if not which or "gen_spectrum" in which:
                # plot MC generator spectrum and selected spectrum
                fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
                for i, key in enumerate(events):
                    ax = axs[i]
                    plt.sca(ax)
                    plt.plot(bin_centres[key].value,
                             SensCalc.generator_energy_hists[key], label="generated",
                             # align="center", width=bin_widths[key].value
                             )
                    plt.plot(bin_centres[key].value,
                             SensCalc.selected_events[key], label="selected",
                             # align="center", width=bin_widths[key].value
                             )
                    plt.xlabel(
                        r"$E_\mathrm{MC} / \mathrm{" + str(bin_centres[key].unit) + "}$")
                    if i == 0:
                        plt.ylabel("number of (unweighted) events")
                    plt.gca().set_xscale("log")
                    plt.gca().set_yscale("log")
                    plt.title(channel_map[key])
                    plt.legend()
                plt.suptitle(mode)
                plt.subplots_adjust(left=.1, wspace=.1)
                if args.write:
                    save_fig(args.plots_dir + "generator_events_" + mode)
                plt.pause(.1)

            if not which or "expected_events" in which:
                # plot the number of expected events in each energy bin
                plt.figure()
                for key in ['g', 'p', 'e']:
                    plt.plot(
                        bin_centres[key] / energy_unit,
                        SensCalc.exp_events_per_energy_bin[key],
                        label=channel_map[key],
                        color=channel_color_map[key],
                        # align="center", width=bin_widths[key].value, alpha=.75
                    )
                plt.gca().set_xscale("log")
                plt.gca().set_yscale("log")
                plt.title(mode)

                plt.xlabel(r"$E_\mathrm{MC} / \mathrm{" + str(energy_unit) + "}$")
                plt.ylabel("expected events in {}".format(observation_time))
                plt.legend()
                if args.write:
                    save_fig(args.plots_dir + "expected_events_" + mode)
                plt.pause(.1)

            if not which or "event_rate" in which:
                # plot the number of expected events in each energy bin
                plt.figure()
                for key in ['p', 'e', 'g']:
                    plt.plot(
                        bin_centres[key] / energy_unit,
                        (SensCalc.exp_events_per_reco_energy_bin[key] /
                         observation_time).to(u.s**-1).value *
                        (1 if key == 'g' else alpha),
                        label=channel_map[key],
                        marker=channel_marker_map[key],
                        color=channel_color_map[key],
                        # align="center", width=bin_widths[key].value, alpha=.75
                    )
                plt.gca().set_xscale("log")
                plt.gca().set_yscale("log")

                plt.xlabel(r"$E_\mathrm{MC} / \mathrm{" + str(energy_unit) + "}$")
                plt.ylabel(r"event rate: $\frac{dN}{dt} / \mathrm{s}^{-1}$")
                plt.title(mode)
                plt.legend()
                if args.write:
                    save_fig(args.plots_dir + "event_rate_" + mode)
                plt.pause(.1)

            if not which or "effective_areas" in which:
                # plot effective area
                plt.figure()
                plt.suptitle("Effective Areas")
                for key in ['g', 'p', 'e']:
                    plt.plot(
                        bin_centres[key] / energy_unit,
                        SensCalc.effective_areas[key] / u.m**2,
                        label=channel_map[key], color=channel_color_map[key],
                        marker=channel_marker_map[key])
                plt.xlabel(r"$E_\mathrm{MC} / \mathrm{" + str(energy_unit) + "}$")
                plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
                plt.gca().set_xscale("log")
                plt.gca().set_yscale("log")
                plt.title(mode)
                plt.legend()
                if args.write:
                    save_fig(args.plots_dir + "effective_areas_" + mode)
                plt.pause(.1)


# ##     ##    ###    #### ##    ##
# ###   ###   ## ##    ##  ###   ##
# #### ####  ##   ##   ##  ####  ##
# ## ### ## ##     ##  ##  ## ## ##
# ##     ## #########  ##  ##  ####
# ##     ## ##     ##  ##  ##   ###
# ##     ## ##     ## #### ##    ##

if __name__ == "__main__":
    np.random.seed(19)

    parser = make_argparser()
    parser.add_argument('--infile', type=str, default="classified_events")
    parser.add_argument('--load', action="store_true", default=False,
                        help="load splines instead of fitting and writing")

    args = parser.parse_args()

    # load meta data from disk
    meta_data_file = f"{args.indir}/meta_data.yml"
    meta_data = yaml.load(open(meta_data_file), Loader=Loader)
    meta_units = meta_data["units"]
    meta_gammas = meta_data["gamma"]
    meta_proton = meta_data["proton"]
    meta_electr = meta_data["electron"]

    for meta in [meta_gammas, meta_proton, meta_electr]:
        meta["n_simulated"] = meta["n_files"] * meta["n_events_per_file"]

    # reading the reconstructed and classified events
    gammas_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
        args.indir, args.infile, "gamma", "wave"), "reco_events")
    proton_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
        args.indir, args.infile, "proton", "wave"), "reco_events")
    electr_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
        args.indir, args.infile, "electron", "wave"), "reco_events")

    # FUCK FUCK FUCK FUCK
    correct_off_angle(gammas_w_o)
    correct_off_angle(proton_w_o)
    correct_off_angle(electr_w_o)

    events_w = {"reco": {'g': gammas_w_o, 'p': proton_w_o, 'e': electr_w_o}}

    gammas_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
        args.indir, args.infile, "gamma", "tail"), "reco_events")
    proton_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
        args.indir, args.infile, "proton", "tail"), "reco_events")
    electr_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
        args.indir, args.infile, "electron", "tail"), "reco_events")

    events_t = {"reco": {'g': gammas_t_o, 'p': proton_t_o, 'e': electr_t_o}}

    if args.load:
        # print("reading pickled splines")
        # from sklearn.externals import joblib
        # spline_w_ga = joblib.load("./data/spline_wave_gammaness.pkl")
        # spline_w_th = joblib.load("./data/spline_wave_th.pkl")
        # spline_t_ga = joblib.load("./data/spline_tail_gammaness.pkl")
        # spline_t_th = joblib.load("./data/spline_tail_th.pkl")

        print("loading cut values")
        from astropy.table import Table
        cut_energies, ga_cuts, th_cuts = {}, {}, {}
        spline_ga, spline_th = {}, {}
        for mode in ["wave", "tail"]:
            cuts = Table.read(f"scripts/cut_values_{mode}.tex", format="ascii.latex")
            cut_energies[mode] = cuts["Energy"]
            ga_cuts[mode] = cuts["gammaness"]
            th_cuts[mode] = cuts["theta"]

            spline_ga[mode] = interpolate.splrep(cut_energies[mode],
                                                 ga_cuts[mode], k=1)
            spline_th[mode] = interpolate.splrep(cut_energies[mode],
                                                 th_cuts[mode], k=1)
        spline_w_ga = spline_ga["wave"]
        spline_t_ga = spline_ga["tail"]
        spline_w_th = spline_th["wave"]
        spline_t_th = spline_th["tail"]

    else:
        print("making splines")
        cut_energies = sensitivity_energy_bin_edges[::]
        cut_energies_mid = np.sqrt(cut_energies[:-1] * cut_energies[1:])
        (spline_w_ga, ga_cuts_w), (spline_w_th, th_cuts_w) = \
            get_optimal_splines(events_w["reco"], cut_energies, k=1)
        print("... wavelets done")
        (spline_t_ga, ga_cuts_t), (spline_t_th, th_cuts_t) = \
            (spline_w_ga, ga_cuts_w), (spline_w_th, xi_cuts_w)
        # get_optimal_splines(events_t["reco"], cut_energies, k=1)
        print("... tailcuts done")

        print("writing pickled splines")
        from sklearn.externals import joblib
        joblib.dump(spline_w_ga, "./data/spline_wave_gammaness.pkl")
        joblib.dump(spline_w_th, "./data/spline_wave_th.pkl")
        joblib.dump(spline_t_ga, "./data/spline_tail_gammaness.pkl")
        joblib.dump(spline_t_th, "./data/spline_tail_th.pkl")

    if False:
        # wave
        fig = plt.figure(figsize=(10, 5))
        fig.add_subplot(121)
        if not args.load:
            plt.plot(cut_energies_mid / u.TeV, ga_cuts_w,
                     label="crit. values", ls="", marker="^")
        plt.plot(e_bin_fine_edges / u.TeV,
                 interpolate.splev(e_bin_fine_edges, spline_w_ga),
                 label="spline fit")

        plt.xlabel("Energy / TeV")
        plt.ylabel("gammaness")
        plt.gca().set_xscale("log")
        plt.legend()

        fig.add_subplot(122)
        if not args.load:
            plt.plot(cut_energies_mid / u.TeV, th_cuts_w,
                     label="crit. values", ls="", marker="^")
        plt.plot(e_bin_fine_edges / u.TeV,
                 interpolate.splev(e_bin_fine_edges, spline_w_th),
                 label="spline fit")
        plt.xlabel("Energy / TeV")
        plt.ylabel("xi / degree")
        plt.gca().set_xscale("log")
        plt.legend()

        plt.suptitle("wavelets")

        if args.write:
            save_fig(args.plots_dir + "cuts_vs_E_wave")

        plt.pause(.1)

        # tail
        fig = plt.figure(figsize=(10, 5))
        fig.add_subplot(121)
        if not args.load:
            plt.plot(cut_energies_mid / u.TeV, ga_cuts_t,
                     label="crit. values", ls="", marker="^")
        plt.plot(e_bin_fine_edges, interpolate.splev(e_bin_fine_edges, spline_t_ga),
                 label="spline fit")

        plt.xlabel("Energy / TeV")
        plt.ylabel("gammaness")
        plt.gca().set_xscale("log")
        plt.legend()

        fig.add_subplot(122)
        if not args.load:
            plt.plot(cut_energies_mid / u.TeV, th_cuts_t,
                     label="crit. values", ls="", marker="^")
        plt.plot(e_bin_fine_edges, interpolate.splev(e_bin_fine_edges, spline_t_th),
                 label="spline fit")
        plt.xlabel("Energy / TeV")
        plt.ylabel("xi / degree")
        plt.gca().set_xscale("log")
        plt.legend()

        plt.suptitle("tailcuts")

        if args.write:
            save_fig(args.plots_dir + "cuts_vs_E_tail")

        plt.pause(.1)
        plt.show()
    # end load splines

    from_step = "reco"
    next_step = "gammaness"
    events_w[next_step] = {}
    events_t[next_step] = {}
    for key in events_w[from_step]:
        events_w[next_step][key] = events_w[from_step][key][
            (events_w[from_step][key]["gammaness"] >
             interpolate.splev(events_w[from_step][key]["reco_Energy"], spline_w_ga))]
        events_t[next_step][key] = events_t[from_step][key][
            (events_t[from_step][key]["gammaness"] >
             interpolate.splev(events_t[from_step][key]["reco_Energy"], spline_t_ga))]

    from_step = "gammaness"
    next_step = "theta"
    events_w[next_step] = {}
    events_t[next_step] = {}
    for key in events_w[from_step]:
        events_w[next_step][key] = events_w[from_step][key][
            (events_w[from_step][key]["off_angle"] < (1 if key == 'g' else r_scale) *
             interpolate.splev(events_w[from_step][key]["reco_Energy"], spline_w_th))]
        events_t[next_step][key] = events_t[from_step][key][
            (events_t[from_step][key]["off_angle"] < (1 if key == 'g' else r_scale) *
             interpolate.splev(events_t[from_step][key]["reco_Energy"], spline_t_th))]

    for step in ["reco", "theta"]:
        ev = events_w[step]
        print(f"selected events at step {step}:")
        for ch in ['g', 'p', 'e']:
            print(f"{ch}: {len(ev[ch])}")

    plots_dir_temp = args.plots_dir
    # for step in ["theta"]:  # "reco", "gammaness", "theta"]:
    for step in ["gammaness"]:  # "reco", "gammaness", "theta"]:
        args.plots_dir = "/".join([plots_dir_temp, step, ""])
        if not os.path.exists(args.plots_dir):
            os.makedirs(args.plots_dir)
        make_performance_plots(events_w[step],
                               events_t[step],
                               which=["theta_square", "ang_res"])

    # plt.show()

    # plt.figure()
    # ax1 = plt.subplot(211)
    # xi_68_w = percentiles(events_w["gammaness"]['g']["off_angle"],
    #                       events_w["gammaness"]['g']["reco_Energy"],
    #                       e_bin_edges.value, 68)
    # xi_68_t = percentiles(events_t["gammaness"]['g']["off_angle"],
    #                       events_t["gammaness"]['g']["reco_Energy"],
    #                       e_bin_edges.value, 68)
    # plt.plot(e_bin_centres.value, xi_68_t / xi_68_w)
    # plt.plot(e_bin_centres.value, np.ones_like(e_bin_centres.value),
    #          ls="dashed", color="gray")
    # plt.xlabel(r"$\log_{10}(E_\mathrm{reco}$ / TeV)")
    # plt.ylabel("ratio")
    # ax1 = plt.subplot(212)
    # plt.plot([0,1], [0,1])
    # save_fig("./ang_res_ratio")
    # plt.show()

    sens_w = calculate_sensitivities(
        events_w["theta"], sensitivity_energy_bin_edges, alpha=alpha, n_draws=1)
    sens_t = calculate_sensitivities(
        events_t["theta"], sensitivity_energy_bin_edges, alpha=alpha, n_draws=1)

    make_sensitivity_plots(sens_w, sens_w.sensitivities,
                           sens_t, sens_t.sensitivities)

    if args.plot:
        plt.show()

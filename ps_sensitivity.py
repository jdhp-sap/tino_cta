#!/usr/bin/env python3
import glob
import numpy as np

# PyTables
import tables as tb
# pandas data frames
import pandas as pd

from astropy.table import Table
from astropy import units as u

from itertools import chain

from helper_functions import *

from ctapipe.analysis.sensitivity import (SensitivityPointSource, e_minus_2,
                                          crab_source_rate, cr_background_rate)

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
# plt.style.use('t_slides')


# MC energy ranges:
# gammas: 0.1 to 330 TeV
# proton: 0.1 to 600 TeV
edges_gammas = np.logspace(2, np.log10(330000), 28) * u.GeV
edges_proton = np.logspace(2, np.log10(600000), 30) * u.GeV

# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg*u.cm**2*u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

# scale MC events to this reference time
observation_time = 50*u.h


def open_pytable_as_pandas(filename, mode='r'):
    pyt_infile = tb.open_file(filename, mode=mode)
    pyt_table = pyt_infile.root.reco_events

    return pd.DataFrame(pyt_table[:])


def main_const_theta_cut():

    def selection_mask(event_table, ntels=3, gammaness=.75, r_max=0.1*u.deg):
        return ((event_table["NTels_reco"] >= ntels) &
                (event_table["gammaness"] > gammaness) &
                (event_table["off_angle"] < r_max))

    apply_cuts = True
    gammaness_wave = .75
    gammaness_tail = .75
    r_max_gamm_wave = 0.04*u.deg
    r_max_gamm_tail = 0.04*u.deg
    r_max_prot = 2*u.deg

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * (498-14)
    NProton_simulated = NProton_per_File * (6998-100)

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    gammas = open_pytable_as_pandas(
            "{}/{}_{}_{}_run1001-run1012.h5".format(
                    args.events_dir, args.in_file, "gamma", "wave"))

    proton = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10000-run10043.h5".format(
                    args.events_dir, args.in_file, "proton", "wave"))

    print()
    print("gammas present (wavelets):", len(gammas))
    print("proton present (wavelets):", len(proton))

    # applying some cuts
    if apply_cuts:
        gammas = gammas[selection_mask(
                gammas, gammaness=gammaness_wave, r_max=r_max_gamm_wave)]
        proton = proton[selection_mask(
                proton, gammaness=gammaness_wave, r_max=r_max_prot)]

    print()
    print("gammas selected (wavelets):", len(gammas))
    print("proton selected (wavelets):", len(proton))

    SensCalc = SensitivityPointSource(
                reco_energies={'g': gammas['MC_Energy'].values*u.GeV,
                               'p': np.random.uniform(100, 600000, len(proton))*u.GeV},
                mc_energies={'g': gammas['MC_Energy'].values*u.GeV,
                             'p': proton['MC_Energy'].values*u.GeV},
                energy_bin_edges={'g': edges_gammas,
                                  'p': edges_proton},
                flux_unit=flux_unit)

    sensitivities = SensCalc.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': e_minus_2, 'p': e_minus_2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2},
                            alpha=(r_max_gamm_wave/r_max_prot)**2,
                            # sensitivity_energy_bin_edges=
                            #     10**np.array([-1, -.75, -.5, -.25, 0, 2,
                            #               2.25, 2.5, 2.75, 3, 9])*u.TeV
                                )
    weights = SensCalc.event_weights

    NExpGammas = sum(SensCalc.exp_events_per_energy_bin['g'])
    NExpProton = sum(SensCalc.exp_events_per_energy_bin['p'])

    print()
    print("expected gammas (wavelets):", NExpGammas)
    print("expected proton (wavelets):", NExpProton)

    # now for tailcut
    gammas_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run1015-run1026.h5".format(
                    args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10100-run10143.h5".format(
                    args.events_dir, args.in_file, "proton", "tail"))

    if False:
        fig = plt.figure()
        tax = plt.subplot(121)
        histo = np.histogram2d(gammas_t["NTels_reco"], gammas_t["gammaness"],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        im = tax.imshow(histo_normed, interpolation='none', origin='lower',
                        aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
        cb = fig.colorbar(im, ax=tax)
        tax.set_title("gammas")
        tax.set_xlabel("NTels")
        tax.set_ylabel("gammaness")

        tax = plt.subplot(122)
        histo = np.histogram2d(proton_t["NTels_reco"], proton_t["gammaness"],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        im = tax.imshow(histo_normed, interpolation='none', origin='lower',
                        aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
        cb = fig.colorbar(im, ax=tax)
        tax.set_title("protons")
        tax.set_xlabel("NTels")
        tax.set_ylabel("gammaness")

        plt.show()

    print()
    print("gammas present (tailcuts):", len(gammas_t))
    print("proton present (tailcuts):", len(proton_t))

    # applying some cuts
    if apply_cuts:
        gammas_t = gammas_t[selection_mask(
                gammas_t, gammaness=gammaness_tail, r_max=r_max_gamm_tail)]
        proton_t = proton_t[selection_mask(
                proton_t, gammaness=gammaness_tail, r_max=r_max_prot)]

    print()
    print("gammas selected (tailcuts):", len(gammas_t))
    print("proton selected (tailcuts):", len(proton_t))

    SensCalc_t = SensitivityPointSource(
                reco_energies={'g': gammas_t['MC_Energy'].values*u.GeV,
                               'p': np.random.uniform(100, 600000, len(proton_t))*u.GeV},
                mc_energies={'g': gammas_t['MC_Energy'].values*u.GeV,
                             'p': proton_t['MC_Energy'].values*u.GeV},
                energy_bin_edges={'g': edges_gammas,
                                  'p': edges_proton},
                flux_unit=flux_unit)

    sensitivities_t = SensCalc_t.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': e_minus_2, 'p': e_minus_2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2},
                            alpha=(r_max_gamm_tail/r_max_prot)**2)

    weights_t = SensCalc_t.event_weights

    gammas_t["weights"] = weights_t['g']
    proton_t["weights"] = weights_t['p']

    NExpGammas_t = sum(SensCalc_t.exp_events_per_energy_bin['g'])
    NExpProton_t = sum(SensCalc_t.exp_events_per_energy_bin['p'])

    print()
    print("expected gammas (tailcuts):", NExpGammas_t)
    print("expected proton (tailcuts):", NExpProton_t)

    # do some plotting
    if args.plot:
        make_sensitivity_plots(edges_gammas, edges_proton, SensCalc, SensCalc_t,
                               sensitivities, sensitivities_t)


def main_xi68_cut():

    def selection_mask(event_table, ntels=3, gammaness=.75, r_max=None):
        return ((event_table["NTels_reco"] >= ntels) &
                (event_table["gammaness"] > gammaness) &
                (event_table["off_angle"] < r_max(event_table["reco_Energy"])))

    apply_cuts = True
    gammaness_wave = .75
    gammaness_tail = .75
    theta_on_off_ratio = 4.5

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * (498-14)
    NProton_simulated = NProton_per_File * (6998-100)

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    gammas = open_pytable_as_pandas(
            "{}/{}_{}_{}_run1001-run1012.h5".format(
                    args.events_dir, args.in_file, "gamma", "wave"))

    proton = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10000-run10043.h5".format(
                    args.events_dir, args.in_file, "proton", "wave"))

    print()
    print("gammas present (wavelets):", len(gammas))
    print("proton present (wavelets):", len(proton))

    # now for tailcut
    gammas_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run1015-run1026.h5".format(
                    args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10100-run10143.h5".format(
                    args.events_dir, args.in_file, "proton", "tail"))
    print()
    print("gammas present (tailcuts):", len(gammas_t))
    print("proton present (tailcuts):", len(proton_t))

    # faking reconstructed energy
    gammas["reco_Energy"] = gammas["MC_Energy"] * u.GeV.to(u.TeV)
    proton["reco_Energy"] = np.random.uniform(100, 600000, len(proton))*u.GeV
    gammas_t["reco_Energy"] = gammas_t["MC_Energy"] * u.GeV.to(u.TeV)
    proton_t["reco_Energy"] = np.random.uniform(100, 600000, len(proton_t))*u.GeV

    # define edges to sort events in
    n_e_bins = 20
    e_bins_fine = np.logspace(-1, np.log10(600), n_e_bins)*u.TeV
    xi_ebinned_w = [[] for a in range(n_e_bins)]
    xi_ebinned_t = [[] for a in range(n_e_bins)]

    for xi, en in zip(gammas["off_angle"], gammas["reco_Energy"]):
        xi_ebinned_w[np.digitize(en, e_bins_fine)].append(xi)
    for xi, en in zip(gammas_t["off_angle"], gammas_t["reco_Energy"]):
        xi_ebinned_t[np.digitize(en, e_bins_fine)].append(xi)

    # get the 68th percentile resolution in every energy bin
    xi68_ebinned_w = np.full(len(xi_ebinned_w), np.inf)
    xi68_ebinned_t = np.full(len(xi_ebinned_t), np.inf)
    for i, (ebin_w, ebin_t) in enumerate(zip(xi_ebinned_w, xi_ebinned_t)):
        try:
            xi68_ebinned_w[i] = np.percentile(ebin_w, 68)
            xi68_ebinned_t[i] = np.percentile(ebin_t, 68)
        except IndexError:
            pass

    from scipy.optimize import curve_fit
    popt_w, pcov_w = curve_fit(xi_fitfunc, e_bins_fine[1:-1].value, xi68_ebinned_w[1:-1])
    popt_t, pcov_t = curve_fit(xi_fitfunc, e_bins_fine[1:-1].value, xi68_ebinned_t[1:-1])

    if False:
        plt.figure()
        plt.semilogx(e_bins_fine[1:-1], xi68_ebinned_w[1:-1],
                     color="darkred", label="MC wave")
        plt.semilogx(e_bins_fine[1:-1], xi68_ebinned_t[1:-1],
                     color="darkorange", label="MC tail")
        plt.semilogx(e_bins_fine[1:-1], xi_fitfunc(e_bins_fine[1:-1].value, *popt_w),
                     marker="^", ls="", color="darkred", label="fit wave")
        plt.semilogx(e_bins_fine[1:-1], xi_fitfunc(e_bins_fine[1:-1].value, *popt_t),
                     marker="^", ls="", color="darkorange", label="fit tail")
        plt.xlabel("E / TeV")
        plt.ylabel(r"$\xi_{68}$ / deg")
        plt.legend()
        plt.pause(.1)

    # applying some cuts
    if apply_cuts:
        gammas = gammas[selection_mask(
                gammas, gammaness=gammaness_wave,
                r_max=lambda e: xi_fitfunc(e, *popt_w))]
        proton = proton[selection_mask(
                proton, gammaness=gammaness_wave,
                r_max=lambda e: xi_fitfunc(e, *popt_w)*theta_on_off_ratio)]
        gammas_t = gammas_t[selection_mask(
                gammas_t, gammaness=gammaness_wave,
                r_max=lambda e: xi_fitfunc(e, *popt_t))]
        proton_t = proton_t[selection_mask(
                proton_t, gammaness=gammaness_wave,
                r_max=lambda e: xi_fitfunc(e, *popt_t)*theta_on_off_ratio)]

    SensCalc = SensitivityPointSource(
            reco_energies={'g': gammas['reco_Energy'].values*u.TeV,
                           'p': proton['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas['MC_Energy'].values*u.GeV,
                         'p': proton['MC_Energy'].values*u.GeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton},
            flux_unit=flux_unit)

    event_weights = SensCalc.generate_event_weights(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2})

    sensitivities = SensCalc.get_sensitivity(
            alpha=theta_on_off_ratio**-2,
            sensitivity_energy_bin_edges=np.logspace(-1, 3, 17)*u.TeV)

    # sensitvity for tail cuts
    SensCalc_t = SensitivityPointSource(
            reco_energies={'g': gammas_t['reco_Energy'].values*u.TeV,
                           'p': proton_t['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas_t['MC_Energy'].values*u.GeV,
                         'p': proton_t['MC_Energy'].values*u.GeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton},
            flux_unit=flux_unit)

    event_weights_t = SensCalc_t.generate_event_weights(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2})

    sensitivities_t = SensCalc_t.get_sensitivity(
            alpha=theta_on_off_ratio**-2,
            sensitivity_energy_bin_edges=np.logspace(-1, 3, 17)*u.TeV)

    make_sensitivity_plots(edges_gammas, edges_proton, SensCalc, SensCalc_t,
                           sensitivities, sensitivities_t)


def xi_fitfunc(x, a, b, c, d, e, f, g, h):
    x = np.log10(x)
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7


def make_sensitivity_plots(edges_gammas, edges_proton, SensCalc, SensCalc_t,
                           sensitivities, sensitivities_t):
        bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.
        bin_centres_p = (edges_proton[1:]+edges_proton[:-1])/2.

        bin_widths_g = np.diff(edges_gammas.value)
        bin_widths_p = np.diff(edges_proton.value)

        if args.verbose:
            # plot MC generator spectrum and selected spectrum
            plt.figure()
            plt.subplot(121)
            plt.bar(bin_centres_g.value,
                    SensCalc_t.generator_energy_hists['g'], label="generated",
                    align="center", width=bin_widths_g)
            plt.bar(bin_centres_g.value,
                    SensCalc_t.selected_events['g'], label="selected",
                    align="center", width=bin_widths_g)
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel("number of events")
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")
            plt.title("gammas -- tailcuts")
            plt.legend()

            plt.subplot(122)
            plt.bar(bin_centres_p.value,
                    SensCalc_t.generator_energy_hists['p'], label="generated",
                    align="center", width=bin_widths_p)
            plt.bar(bin_centres_p.value,
                    SensCalc_t.selected_events['p'], label="selected",
                    align="center", width=bin_widths_p)
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel("number of events")
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")
            plt.title("protons -- tailcuts")
            plt.legend()

            # plot the number of expected events in each energy bin
            plt.figure()
            plt.bar(
                    bin_centres_p.value,
                    SensCalc_t.exp_events_per_energy_bin['p'], label="proton",
                    align="center", width=np.diff(edges_proton.value), alpha=.75)
            plt.bar(
                    bin_centres_g.value,
                    SensCalc_t.exp_events_per_energy_bin['g'], label="gamma",
                    align="center", width=np.diff(edges_gammas.value), alpha=.75)
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")

            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.legend()

            # plot effective area
            plt.figure(figsize=(16, 8))
            plt.suptitle("ASTRI Effective Areas")
            plt.subplot(121)
            plt.loglog(
                bin_centres_g,
                SensCalc.effective_areas['g'], label="wavelets")
            plt.loglog(
                bin_centres_g,
                SensCalc_t.effective_areas['g'], label="tailcuts")
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel(r"effective area / $\mathrm{m^2}$")
            plt.title("gammas")
            plt.legend()

            plt.subplot(122)
            plt.loglog(
                bin_centres_p,
                SensCalc.effective_areas['p'], label="wavelets")
            plt.loglog(
                bin_centres_p,
                SensCalc_t.effective_areas['p'], label="tailcuts")
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_p.unit)+"}$")
            plt.ylabel(r"effective area / $\mathrm{m^2}$")
            plt.title("protons")
            plt.legend()

            # plot the angular distance of the reconstructed shower direction
            # from the pseudo-source

            figure = plt.figure()
            bins = 60

            plt.subplot(211)
            plt.hist([proton_t['off_angle']**2,
                      gammas_t["off_angle"]**2],
                     weights=[weights_t['p'], weights_t['g']],
                     rwidth=1, stacked=True,
                     range=(0, .3), label=("protons", "gammas"),
                     log=False, bins=bins)
            plt.xlabel(r"$(\vartheta/^\circ)^2$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.xlim([0, .3])
            plt.legend(loc="upper right", title="tailcuts")

            plt.subplot(212)
            plt.hist([proton['off_angle']**2,
                      gammas["off_angle"]**2],
                     weights=[weights['p'], weights['g']],
                     rwidth=1, stacked=True,
                     range=(0, .3), label=("protons", "gammas"),
                     log=False, bins=bins)
            plt.xlabel(r"$(\vartheta/^\circ)^2$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.xlim([0, .3])
            plt.legend(loc="upper right", title="wavelets")
            plt.tight_layout()

            if args.write:
                save_fig("plots/theta_square")

        # the point-source sensitivity binned in energy

        plt.figure()
        # draw the crab flux as well
        crab_bins = np.logspace(-1, 3, 17)
        plt.loglog(crab_bins,
                   (crab_source_rate(crab_bins*u.TeV).to(flux_unit)
                    * (crab_bins*u.TeV.to(u.erg))**2),
                   color="red", ls="dashed", label="Crab Nebula")
        plt.loglog(crab_bins,
                   (crab_source_rate(crab_bins*u.TeV).to(flux_unit)
                    * (crab_bins*u.TeV.to(u.erg))**2)/10,
                   color="red", ls="dashed", alpha=.66, label="Crab Nebula / 10")
        plt.loglog(crab_bins,
                   (crab_source_rate(crab_bins*u.TeV).to(flux_unit)
                    * (crab_bins*u.TeV.to(u.erg))**2)/100,
                   color="red", ls="dashed", alpha=.33, label="Crab Nebula / 100")

        # plt.semilogy(
        #     sensitivities["Energy"],
        #     (sensitivities["Sensitivity"].to(flux_unit) *
        #      sensitivities["Energy"].to(u.erg)**2),
        #     color="darkred",
        #     marker="s",
        #     label="wavelets")
        plt.semilogy(
            sensitivities["Energy"].to(u.TeV),
            (sensitivities["Sensitivity_base"].to(flux_unit) *
             sensitivities["Energy"].to(u.erg)**2),
            color="darkgreen",
            marker="^",
            # ls="",
            label="wavelets (no upscale)")

        # plt.semilogy(
        #     sensitivities_t["Energy"].to(energy_unit),
        #     (sensitivities_t["Sensitivity"].to(flux_unit) *
        #      sensitivities_t["Energy"].to(u.erg)**2),
        #     color="C0",
        #     marker="s",
        #     label="tailcuts")
        plt.semilogy(
            sensitivities_t["Energy"].to(energy_unit),
            (sensitivities_t["Sensitivity_base"].to(flux_unit) *
             sensitivities_t["Energy"].to(u.erg)**2),
            color="darkorange",
            marker="^",
            # ls="",
            label="tailcuts (no upscale)")

        plt.legend(title="Obsetvation Time: {}".format(observation_time))
        plt.xlabel('E / {:latex}'.format(energy_unit))
        plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
        plt.gca().set_xscale("log")
        plt.grid()
        plt.xlim([1e-2, 2e3])
        plt.ylim([5e-14, 5e-9])

        # plot the sensitivity ratios
        plt.figure()
        plt.semilogx(sensitivities_t["Energy"].to(energy_unit),
                     (sensitivities["Sensitivity_base"].to(flux_unit) *
                      sensitivities["Energy"].to(u.erg)**2)[1:] /
                     (sensitivities_t["Sensitivity_base"].to(flux_unit) *
                      sensitivities_t["Energy"].to(u.erg)**2),
                     label=r"Sens$_{wave}$ / Sens$_{tail}$"
                     )
        plt.legend()
        plt.semilogx(sensitivities_t["Energy"].to(energy_unit)[[0, -1]],
                     [1, 1], ls="--", color="gray")
        plt.xlim(sensitivities_t["Energy"].to(energy_unit)[[0, -1]].value)
        plt.ylim([.25, 1.1])
        plt.xlabel('E / {:latex}'.format(energy_unit))
        plt.ylabel("ratio")

        # plot a sky image of the events
        # useless since too few MC background events left
        if False:
            fig2 = plt.figure()
            plt.hexbin(
                [(ph-180)*np.sin(th*u.deg) for
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

        # this demonstrates how to flatten the proton distribution in the theta plot:
        #     NProtons = np.sum(proton['off_angle'][(proton['off_angle'].values**2) < 10])
        #     proton_weight_flat = np.ones(bins) * NProtons/bins
        #     proton_angle_flat = np.linspace(0, 10, bins, False)
        #     proton_angle = proton_angle_flat
        #     proton_weight = proton_weight_flat


if __name__ == "__main__":
    np.random.seed(19)

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/events")
    parser.add_argument('--in_file', type=str, default="classified_events")
    args = parser.parse_args()

    # from itertools import count
    # for i in count(19):
    #     print(i)
    #     np.random.seed(i)
    #     main_xi68_cut()
    #     plt.show()

    main_xi68_cut()
    # main_const_theta_cut()
    if args.plot:
        plt.show()

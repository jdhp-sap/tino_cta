from matplotlib import pyplot as plt

import numpy as np

from astropy.table import Table
from astropy import units as u

from helper_functions import *

# imports the PS module
from ctapipe.analysis.Sensitivity import *
# imports a few spectral flux functions
from ctapipe.analysis.Sensitivity import (crab_source_rate, Eminus2, CR_background_rate,
                                          )

# FIXME need to know:
# • N simulated events
# • simulated energy spectrum
# • generator areas


if __name__ == "__main__":

    edges_gammas = np.logspace(0, np.log10(330000), 38) * u.GeV
    edges_proton = np.logspace(0, np.log10(600000), 40) * u.GeV

    gammas = Table.read("/local/home/tmichael/Data/cta/Tarek/"
                        "CTA_MARS_gamma_3_3HB1-ND.fits")
    proton = Table.read("/local/home/tmichael/Data/cta/Tarek/"
                        "CTA_MARS_proton_3_3HB1-ND.fits")
    electr = Table.read("/local/home/tmichael/Data/cta/Tarek/"
                        "CTA_MARS_electron_3_3HB1-ND.fits")

    print("E min, max gammas:", np.min(gammas['MC_ENERGY']), np.max(gammas['MC_ENERGY']))
    print("E min, max proton:", np.min(proton['MC_ENERGY']), np.max(proton['MC_ENERGY']))
    print("E min, max electr:", np.min(electr['MC_ENERGY']), np.max(electr['MC_ENERGY']))

    gen_spectrum = lambda x: (x/u.GeV) ** (-2) / (u.GeV * u.s * u.m**2)

    NGammas_simulated = 5e6
    Nproton_simulated = 5e6
    Nelectr_simulated = 5e6

    gen_area_g = (5*u.km)**2 * np.pi
    gen_area_p = (5*u.km)**2 * np.pi
    gen_area_g = (5*u.km)**2 * np.pi

    mask_gammas = (gammas['MARS_ALT'] != -1) & (gammas['MARS_AZ'] != -1)
    mask_proton = (proton['MARS_ALT'] != -1) & (proton['MARS_AZ'] != -1)
    mask_electr = (electr['MARS_ALT'] != -1) & (electr['MARS_AZ'] != -1)

    gammas_m = gammas[mask_gammas].to_pandas()
    proton_m = proton[mask_proton].to_pandas()
    electr_m = electr[mask_electr].to_pandas()

    off_angles_gammas = np.sqrt((gammas_m['MARS_AZ']*np.cos(gammas_m['MARS_ALT']))**2 +
                                gammas_m['MARS_ALT']**2)
    off_angles_proton = np.sqrt((proton_m['MARS_AZ']*np.cos(proton_m['MARS_ALT']))**2 +
                                proton_m['MARS_ALT']**2)
    off_angles_electr = np.sqrt((electr_m['MARS_AZ']*np.cos(electr_m['MARS_ALT']))**2 +
                                electr_m['MARS_ALT']**2)

    print("setting up 'Sensitivity_PointSource' object")
    Sens = SensitivityPointSource(
        mc_energies={'g': gammas_m['MC_ENERGY'].values*u.GeV,
                     'p': proton_m['MC_ENERGY'].values*u.GeV,
                     'e': electr_m['MC_ENERGY'].values*u.GeV},
        off_angles={'g': off_angles_gammas,
                    'p': off_angles_proton,
                    'e': off_angles_electr},
        energy_bin_edges={'g': edges_gammas, "p": edges_proton, 'e': edges_gammas},
        energy_unit=u.GeV, flux_unit=(u.erg*u.cm**2*u.s)**(-1))
    print("... done")

    print("calling 'calculate_sensitivities'")
    sensitivities = Sens.calculate_sensitivities(
                    n_simulated_events={'g': NGammas_simulated,
                                        'p': Nproton_simulated,
                                        'e': Nelectr_simulated},

                    generator_spectra={'g': gen_spectrum,
                                       'p': gen_spectrum,
                                       'e': gen_spectrum},


                    spectra={'g': crab_source_rate,
                             'p': CR_background_rate,
                             'e': Eminus2},
                    generator_areas={'g': gen_area_g,
                                     'p': gen_area_p,
                                     'e': gen_area_g},
                    e_min_max={'g': (3, 330000)*u.GeV,
                               'p': (4, 600000)*u.GeV,
                               'e': (4, 330000)*u.GeV},
                    generator_gamma={"g": 2, "p": 2, 'e': 2}
                    )
    print("... done")

    #
    # do some plotting
    plt.figure()
    plt.loglog(
        edges_gammas[:-1],
        Sens.effective_areas['g'])
    plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
    plt.ylabel("effective area")
    plt.suptitle("gammas")

    plt.figure()
    plt.loglog(
        sensitivities["MC Energy"],
        sensitivities["Sensitivity"])
    plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
    plt.ylabel("sensitivity")
    plt.show()

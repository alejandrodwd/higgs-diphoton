import warnings
import numpy as np

def calc_myy(photon_pt, photon_eta, photon_phi, photon_E):
    pt0, pt1 = photon_pt[:, 0], photon_pt[:, 1]
    eta0, eta1 = photon_eta[:, 0], photon_eta[:, 1]
    phi0, phi1 = photon_phi[:, 0], photon_phi[:, 1]
    E0, E1 = photon_E[:, 0], photon_E[:, 1]

    sumpx = pt0 * np.cos(phi0) + pt1 * np.cos(phi1)
    sumpy = pt0 * np.sin(phi0) + pt1 * np.sin(phi1)
    sumpz = pt0 * np.sinh(eta0) + pt1 * np.sinh(eta1)
    sumE = E0 + E1
    mass2 = sumE**2 - (sumpx**2 + sumpy**2 + sumpz**2)

    invalid = ~np.isfinite(mass2) | (mass2 < 0)
    if np.any(invalid):
        warnings.warn("Invalid or negative diphoton mass-squared found; returning NaN for those events.")

    safe_mass2 = np.where(invalid, np.nan, mass2)
    return np.sqrt(safe_mass2) / 1000  # MeV -> GeV

### angles and jet features, pT/m_yy
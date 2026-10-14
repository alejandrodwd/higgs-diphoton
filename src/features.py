import numpy as np

# works array-wide, requires 2+ photons per event
def calc_myy(photon_pt, photon_eta, photon_phi, photon_E):
    pt0, pt1 = photon_pt[:, 0], photon_pt[:, 1]
    eta0, eta1 = photon_eta[:, 0], photon_eta[:, 1]
    phi0, phi1 = photon_phi[:, 0], photon_phi[:, 1]
    E0, E1 = photon_E[:, 0], photon_E[:, 1]

    sumpx = pt0*np.cos(phi0) + pt1*np.cos(phi1)
    sumpy = pt0*np.sin(phi0) + pt1*np.sin(phi1)
    sumpz = pt0*np.sinh(eta0) + pt1*np.sinh(eta1)
    sump2 = sumpx**2 + sumpy**2 + sumpz**2
    sumE = E0 + E1
    return np.sqrt(sumE**2 - sump2) / 1000   # MeV -> GeV

### angles and jet features, pT/m_yy
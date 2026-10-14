JET OVERLAP REMOVAL

import numpy as np
import awkward as ak

def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)

def clean_jets(a):
    jet_ok = (a.jet_pt > 25000) & (abs(a.jet_eta) < 2.5)   # MeV
    # (add the jet_jvt pileup cut here, using the ATLAS notebook's value)

    dr0 = delta_r(a.jet_eta, a.jet_phi, a.photon_eta[:, 0], a.photon_phi[:, 0])
    dr1 = delta_r(a.jet_eta, a.jet_phi, a.photon_eta[:, 1], a.photon_phi[:, 1])
    jet_ok = jet_ok & (dr0 > 0.4) & (dr1 > 0.4)

    return a.jet_pt[jet_ok], a.jet_eta[jet_ok], a.jet_phi[jet_ok]
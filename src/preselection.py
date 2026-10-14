import numpy as np

# Crack in barrel/end-cap region
def cut_photon_eta_transition(photon_eta):
    a0 = np.abs(photon_eta[:, 0])
    a1 = np.abs(photon_eta[:, 1])
    return ((a0 > 1.52) | (a0 < 1.37)) & ((a1 > 1.52) | (a1 < 1.37))

# Diphoton trigger with 35 and 25 GeV thresholds - margin to stay clear of turn-on curve
def cut_photon_pt(photon_pt):
    return (photon_pt[:, 0] > 40000) & (photon_pt[:, 1] > 30000)
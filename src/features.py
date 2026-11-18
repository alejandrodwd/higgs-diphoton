import warnings
import numpy as np
import awkward as ak

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


### angles and jet features

def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def clean_jets(a):
    """Jets with pT > 25 GeV, |eta| < 2.5, the ATLAS JVT pileup cut
    (JVT > 0.59 for pT < 60 GeV, |eta| < 2.4) and overlap removal against
    the two leading photons (dR > 0.4). Returns a boolean jagged mask."""
    jet_ok = (a.jet_pt > 25000) & (abs(a.jet_eta) < 2.5)   # MeV
    pileup_prone = (a.jet_pt < 60000) & (abs(a.jet_eta) < 2.4)
    jet_ok = jet_ok & (~pileup_prone | (a.jet_jvt > 0.59))

    dr0 = delta_r(a.jet_eta, a.jet_phi, a.photon_eta[:, 0], a.photon_phi[:, 0])
    dr1 = delta_r(a.jet_eta, a.jet_phi, a.photon_eta[:, 1], a.photon_phi[:, 1])
    return jet_ok & (dr0 > 0.4) & (dr1 > 0.4)


# Classifier inputs. The diphoton mass is deliberately NOT a feature, so the
# classifier cannot learn the 125 GeV bump; it only sees variables that
# describe *how* the photons look, not their combined mass.
#
# Photon momenta enter only as pT / m_yy. With absolute pT1, pT2 plus the
# angles, a flexible model can rebuild m_yy^2 = 2 pT1 pT2 (cosh deta - cos dphi)
# and carve a fake peak into the background (shown in notebook 04). Dividing by
# m_yy makes that identity carry no mass information (same trick as ATLAS).
# Isolation is kept in absolute GeV for the same reason.
FEATURES = [
    # photon kinematics (scale-free)
    "pt1_m", "pt2_m", "pt_yy_m", "abs_eta1", "abs_eta2", "dphi",
    "abs_costheta", "abs_y_yy",
    # photon identification and isolation (what the tutorial cuts use)
    "tight1", "tight2", "etcone20_1", "etcone20_2", "ptcone30_1", "ptcone30_2",
    # hadronic activity (jets after overlap removal)
    "n_jets", "jet1_pt",
]

# Absolute-pT feature set, used ONLY in notebook 04 to demonstrate mass sculpting.
FEATURES_ABS_PT = ["pt1", "pt2", "pt_yy"] + [f for f in FEATURES if not f.endswith("_m")]


def build_features(a):
    """Flat numpy feature dictionary (one entry per event) from a preselected
    awkward array. Also returns `myy` [GeV], which is NOT a training feature."""
    two = lambda x: (ak.to_numpy(x[:, 0]).astype(np.float64),
                     ak.to_numpy(x[:, 1]).astype(np.float64))

    pt1, pt2 = (v / 1000 for v in two(a.photon_pt))   # GeV
    eta1, eta2 = two(a.photon_eta)
    phi1, phi2 = two(a.photon_phi)
    e1, e2 = (v / 1000 for v in two(a.photon_E))
    tight1, tight2 = two(a.photon_isTightID)
    et20_1, et20_2 = (v / 1000 for v in two(a.photon_etcone20))
    pt30_1, pt30_2 = (v / 1000 for v in two(a.photon_ptcone30))

    dphi = np.abs((phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi)
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
    E = e1 + e2
    y_yy = 0.5 * np.log((E + pz) / (E - pz))

    jets = clean_jets(a)
    jet_pt = a.jet_pt[jets]
    n_jets = ak.to_numpy(ak.num(jet_pt)).astype(np.float64)
    jet1_pt = ak.to_numpy(ak.fill_none(ak.max(jet_pt, axis=1), 0.0)) / 1000

    myy = ak.to_numpy(calc_myy(a.photon_pt, a.photon_eta, a.photon_phi, a.photon_E))
    myy = np.where(myy > 0, myy, np.nan)   # bad events are dropped by the mass window later
    pt_yy = np.hypot(px, py)

    return {
        "pt1_m": pt1 / myy, "pt2_m": pt2 / myy, "pt_yy_m": pt_yy / myy,
        "abs_eta1": np.abs(eta1), "abs_eta2": np.abs(eta2),
        "dphi": dphi,
        # |cos theta*| in the diphoton rest frame (pT-balanced approximation)
        "abs_costheta": np.abs(np.tanh(0.5 * (eta1 - eta2))),
        "abs_y_yy": np.abs(y_yy),
        "tight1": tight1, "tight2": tight2,
        "etcone20_1": et20_1, "etcone20_2": et20_2,     # GeV
        "ptcone30_1": pt30_1, "ptcone30_2": pt30_2,     # GeV
        "n_jets": n_jets, "jet1_pt": jet1_pt,
        "myy": myy,                                     # NOT a training feature
        # absolute momenta, only for the sculpting demonstration
        "pt1": pt1, "pt2": pt2, "pt_yy": pt_yy,
    }


def to_matrix(d, features=FEATURES):
    return np.column_stack([d[f] for f in features])

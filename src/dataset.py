# Loading preselected data / signal MC into flat feature tables, plus the
# train / validation / test split shared by every classifier notebook.
from pathlib import Path

import numpy as np
import uproot
from sklearn.model_selection import train_test_split

from preselection import cut_photon_eta_transition, cut_photon_pt
from features import build_features
from weights import LUMI, mc_weight

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MC_FILE = RAW / "mc_343981.ggH125_gamgam.GamGam.root"

MASS_RANGE = (100.0, 160.0)   # GeV, same window as plot_data / the fits
SEED = 42

TRAIN_PERIOD = "A"
TEST_PERIODS = ["B", "C", "D"]
LUMI_TEST = sum(LUMI[p] for p in TEST_PERIODS)   # 9.5 fb^-1

BRANCHES = [
    "photon_n", "trigP", "photon_pt", "photon_eta", "photon_phi",
    "photon_E", "photon_isTightID", "photon_ptcone30", "photon_etcone20",
    "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_jvt",
]
MC_BRANCHES = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_PHOTON",
               "scaleFactor_PhotonTRIGGER", "XSection", "SumWeights"]


def _in_mass_range(d):
    m = d["myy"]
    keep = np.isfinite(m) & (m > MASS_RANGE[0]) & (m < MASS_RANGE[1])
    return {k: v[keep] for k, v in d.items()}


def _cached(name, build):
    path = PROCESSED / f"{name}_features.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    d = build()
    np.savez(path, **d)
    return d


def load_data(period):
    """Preselected data from notebook 01, as features, in the 100-160 GeV window."""
    def build():
        a = uproot.open(f"{PROCESSED}/data_{period}_preselection.root:mini").arrays(BRANCHES)
        return _in_mass_range(build_features(a))
    return _cached(f"data_{period}", build)


def load_signal(lumi=1.0):
    """ggH -> yy MC with the same preselection as the data (notebook 01).
    Weights are normalised to `lumi` fb^-1."""
    def build():
        a = uproot.open(f"{MC_FILE}:mini").arrays(BRANCHES + MC_BRANCHES)
        a = a[a.photon_n >= 2]
        a = a[cut_photon_eta_transition(a.photon_eta)]
        a = a[cut_photon_pt(a.photon_pt)]
        d = build_features(a)
        d["w"] = mc_weight(a, lumi=1.0)
        return _in_mass_range(d)
    d = _cached("mc_ggH125", build)
    d["w"] = d["w"] * lumi
    return d


def concat(dicts):
    return {k: np.concatenate([d[k] for d in dicts]) for k in dicts[0]}


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def load_all():
    """Everything the classifier notebooks need, with a fixed split:

    - signal MC: 50% test; of the other 50%, 80% train / 20% validation
    - data A (background): 80% train / 20% validation
    - data B, C, D: test only (never seen during training or cut choice)

    Signal weights are normalised to the B+C+D luminosity (9.5 fb^-1) *as if the
    split subset were the full sample*, i.e. scaled up by 1/fraction, so that
    yields in any subset are directly comparable to data B+C+D.
    """
    sig = load_signal(lumi=LUMI_TEST)
    bkg_A = load_data(TRAIN_PERIOD)
    data = {p: load_data(p) for p in TEST_PERIODS}

    i_tr, i_te = train_test_split(np.arange(len(sig["w"])), test_size=0.5, random_state=SEED)
    i_tr, i_va = train_test_split(i_tr, test_size=0.2, random_state=SEED)
    j_tr, j_va = train_test_split(np.arange(len(bkg_A["myy"])), test_size=0.2, random_state=SEED)

    n = len(sig["w"])
    out = {
        "sig_train": subset(sig, i_tr), "sig_val": subset(sig, i_va), "sig_test": subset(sig, i_te),
        "bkg_train": subset(bkg_A, j_tr), "bkg_val": subset(bkg_A, j_va),
        "data": data, "data_bcd": concat(list(data.values())),
    }
    for key, idx in [("sig_train", i_tr), ("sig_val", i_va), ("sig_test", i_te)]:
        out[key]["w"] = out[key]["w"] * n / len(idx)
    # data A validation -> expected B+C+D background: scale by luminosity and split fraction
    out["bkg_val_scale"] = LUMI_TEST / LUMI[TRAIN_PERIOD] * len(bkg_A["myy"]) / len(j_va)
    return out

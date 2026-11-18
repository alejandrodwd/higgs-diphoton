# Shared helpers for the classifier notebooks (03 logistic regression,
# 04 boosted tree) so both are trained and evaluated in exactly the same way.
import json
from pathlib import Path

import numpy as np

from features import FEATURES, to_matrix
from weights import balanced
from evaluate import (roc, choose_threshold, in_window, asimov_Z, MyyFit,
                      fit_signal_shape, sig_eff_at_bkg_eff, bkg_rej_at_sig_eff)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
SCORE_DIR = ROOT / "results" / "scores"


def training_set(sig, bkg, features=FEATURES):
    """X, y and class-balanced weights. Signal = MC (label 1), bkg = data A (label 0)."""
    X = np.vstack([to_matrix(sig, features), to_matrix(bkg, features)])
    y = np.r_[np.ones(len(sig["w"])), np.zeros(len(bkg["myy"]))]
    w = balanced(y, np.r_[sig["w"], np.ones(len(bkg["myy"]))])
    return X, y, w


def score_all(model, D, features=FEATURES):
    """Classifier score for every sample used downstream."""
    keys = ["sig_val", "bkg_val", "sig_test", "data_bcd"]
    out = {k: model.predict_proba(to_matrix(D[k], features))[:, 1] for k in keys}
    for p, d in D["data"].items():
        out[f"data_{p}"] = model.predict_proba(to_matrix(d, features))[:, 1]
    return out


def evaluate(scores, D, sigma):
    """Everything we quote for a classifier:
    - ROC/AUC on held-out MC vs data B+C+D
    - score cut chosen on the validation sets only
    - signal efficiency / background rejection of that cut
    - fit of m_yy in data B+C+D after the cut
    """
    s, d = D["sig_test"], D["data_bcd"]
    fpr, tpr, _, auc = roc(scores["sig_test"], s["w"], scores["data_bcd"])

    cut, scan = choose_threshold(scores["sig_val"], D["sig_val"]["w"], D["sig_val"]["myy"],
                                 scores["bkg_val"], D["bkg_val"]["myy"], D["bkg_val_scale"])
    pass_s = scores["sig_test"] >= cut
    pass_d = scores["data_bcd"] >= cut

    fit = MyyFit(d["myy"][pass_d], sigma)                     # mass free -> m_H
    fit125 = MyyFit(d["myy"][pass_d], sigma, fix_mass=125.0)  # mass fixed -> significance
    S_exp = s["w"][pass_s].sum()                              # expected Higgs events, 100-160 GeV
    S_win = s["w"][pass_s & in_window(s["myy"])].sum()

    res = {
        "AUC": auc,
        "sig_eff_at_bkg_eff_1pct": sig_eff_at_bkg_eff(fpr, tpr, 0.01),
        "sig_eff_at_bkg_eff_5pct": sig_eff_at_bkg_eff(fpr, tpr, 0.05),
        "bkg_rej_at_sig_eff_80pct": bkg_rej_at_sig_eff(fpr, tpr, 0.80),
        "cut": float(cut),
        "sig_eff": float(np.abs(s["w"][pass_s]).sum() / np.abs(s["w"]).sum()),
        "bkg_rej": float(1 - pass_d.mean()),
        "N_data": int(pass_d.sum()),
        "S_expected": float(S_exp),
        "N_sig_fit": float(fit125.nsig), "N_sig_err": float(fit125.nsig_err),
        "mu": float(fit125.nsig / S_exp), "mu_err": float(fit125.nsig_err / S_exp),
        "m_H": float(fit.mass), "m_H_err": float(fit.mass_err),
        "Z_obs": float(fit125.Z_local),
        "Z_exp": float(asimov_Z(S_win, fit.background_in_window())),
        "corr_score_myy_bkg": float(np.corrcoef(scores["data_bcd"], d["myy"])[0, 1]),
    }
    return res, {"fpr": fpr, "tpr": tpr, "scan": scan, "fit": fit, "fit125": fit125,
                 "pass_d": pass_d}


def print_summary(name, r):
    print(f"--- {name} ---")
    print(f"AUC (MC test vs data B+C+D)          {r['AUC']:.4f}")
    print(f"sig. eff. @ 1% / 5% bkg eff.          {r['sig_eff_at_bkg_eff_1pct']:.3f} / {r['sig_eff_at_bkg_eff_5pct']:.3f}")
    print(f"bkg. rejection @ 80% sig. eff.        {r['bkg_rej_at_sig_eff_80pct']:.4f}")
    print(f"score cut (chosen on validation)      {r['cut']:.3f}")
    print(f"  signal efficiency / bkg rejection   {r['sig_eff']:.3f} / {r['bkg_rej']:.4f}")
    print(f"  data B+C+D events passing           {r['N_data']}")
    print(f"  expected Higgs events (MC)          {r['S_expected']:.0f}")
    print(f"  fitted Higgs events (m_H = 125)     {r['N_sig_fit']:.0f} ± {r['N_sig_err']:.0f}")
    print(f"  signal strength mu                  {r['mu']:.2f} ± {r['mu_err']:.2f}")
    print(f"  fitted m_H                          {r['m_H']:.2f} ± {r['m_H_err']:.2f} GeV")
    print(f"  local significance obs. / exp.      {r['Z_obs']:.2f} / {r['Z_exp']:.2f} sigma")
    print(f"  corr(score, m_yy) in data           {r['corr_score_myy_bkg']:+.4f}")


def save(name, scores, res):
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORE_DIR / f"{name}_scores.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()})
    with open(SCORE_DIR / f"{name}_summary.json", "w") as f:
        json.dump(res, f, indent=2)

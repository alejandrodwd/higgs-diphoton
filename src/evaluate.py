# ROC/AUC, signal efficiency at fixed background, S/sqrt(B), fits
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from iminuit import Minuit
from sklearn.metrics import roc_auc_score, roc_curve

MASS_RANGE = (100.0, 160.0)
BIN_WIDTH = 1.0   # GeV
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + BIN_WIDTH / 2, BIN_WIDTH)
CENTRES = 0.5 * (BINS[1:] + BINS[:-1])
WINDOW = (120.0, 130.0)   # signal window for cut-and-count numbers


# ---------------------------------------------------------------------------
# classifier metrics (MC signal vs data background)
# ---------------------------------------------------------------------------

def roc(score_sig, w_sig, score_bkg):
    """ROC with signal weighted by |MC weight| and background unit-weighted."""
    y = np.r_[np.ones(len(score_sig)), np.zeros(len(score_bkg))]
    s = np.r_[score_sig, score_bkg]
    w = np.r_[np.abs(w_sig) / np.abs(w_sig).sum(), np.ones(len(score_bkg)) / len(score_bkg)]
    fpr, tpr, thr = roc_curve(y, s, sample_weight=w)
    return fpr, tpr, thr, roc_auc_score(y, s, sample_weight=w)


def sig_eff_at_bkg_eff(fpr, tpr, bkg_eff):
    return float(np.interp(bkg_eff, fpr, tpr))


def bkg_rej_at_sig_eff(fpr, tpr, sig_eff):
    """Background rejection (1 - background efficiency) at a given signal efficiency."""
    return float(1 - np.interp(sig_eff, tpr, fpr))


def asimov_Z(S, B):
    """Median expected discovery significance for S signal on B background."""
    S, B = np.asarray(S, float), np.asarray(B, float)
    return np.sqrt(2 * ((S + B) * np.log1p(S / B) - S))


def in_window(m, window=WINDOW):
    return (m >= window[0]) & (m < window[1])


def choose_threshold(score_sig, w_sig, myy_sig, score_bkg, myy_bkg, bkg_scale,
                     n_steps=200, min_bkg=20):
    """Pick the score cut maximising the expected Asimov significance in the
    120-130 GeV window, using ONLY validation samples (MC val + data A val).
    `bkg_scale` converts data-A-validation counts into the B+C+D expectation.
    Returns (best_cut, table) with table columns: cut, sig_eff, S, B, Z."""
    s_win, b_win = in_window(myy_sig), in_window(myy_bkg)
    S_tot = w_sig[s_win].sum()
    cuts = np.quantile(score_bkg[b_win], np.linspace(0, 1, n_steps, endpoint=False))
    rows = []
    for c in np.unique(cuts):
        S = w_sig[s_win & (score_sig >= c)].sum()
        nb = np.sum(b_win & (score_bkg >= c))
        if nb < min_bkg:   # not enough validation background to trust the estimate
            continue
        B = nb * bkg_scale
        rows.append((c, S / S_tot, S, B, asimov_Z(S, B)))
    table = np.array(rows)
    return table[np.argmax(table[:, 4]), 0], table


# ---------------------------------------------------------------------------
# m_yy fit: 4th-order polynomial background + Gaussian signal, binned
# Poisson likelihood (same model as plot_data, but with a proper likelihood
# so we get a signal yield, its error and a local significance)
# ---------------------------------------------------------------------------

def _x(m):
    return (m - 130.0) / 30.0   # map 100-160 GeV onto [-1, 1] for a stable fit


def _bkg_density(m, c):
    return np.polyval(c[::-1], _x(m))   # c0 + c1 x + ...


def _gauss_bin_fraction(mu, sigma):
    cdf = 0.5 * (1 + erf((BINS - mu) / (np.sqrt(2) * sigma)))
    return np.diff(cdf)


def fit_signal_shape(myy, w):
    """Gaussian core of the MC signal (iterative 2-sigma clipping),
    used to fix the width in the data fit."""
    mu, sigma = 125.0, 2.0
    for _ in range(10):
        sel = np.abs(myy - mu) < 2 * sigma
        mu = np.average(myy[sel], weights=w[sel])
        # 2-sigma truncation of a Gaussian keeps 77.4% of the variance
        sigma = np.sqrt(np.average((myy[sel] - mu) ** 2, weights=w[sel]) / 0.774)
    return mu, sigma


class MyyFit:
    """Binned Poisson likelihood fit of m_yy in 100-160 GeV.

    background: 4th-order polynomial (5 free parameters)
    signal:     N_sig x Gaussian(mass, sigma), sigma fixed from MC,
                mass free in [118, 132] GeV (or fixed if fix_mass is given)
    """

    def __init__(self, myy, sigma, fix_mass=None, deg=4):
        self.n, _ = np.histogram(myy, bins=BINS)
        self.sigma, self.deg, self.fix_mass = sigma, deg, fix_mass
        self._fit()

    def _expected(self, c, nsig, mass):
        b = _bkg_density(CENTRES, c) * BIN_WIDTH
        return np.clip(b, 1e-9, None) + nsig * _gauss_bin_fraction(mass, self.sigma)

    def _nll(self, c, nsig, mass):
        lam = self._expected(c, nsig, mass)
        return np.sum(lam - self.n * np.log(lam))

    def _minimise(self, with_signal):
        # start from a weighted least-squares polynomial fit
        c0 = np.polyfit(_x(CENTRES), self.n, self.deg,
                        w=1 / np.sqrt(np.maximum(self.n, 1)))[::-1] / BIN_WIDTH
        mass0 = self.fix_mass or 125.0
        f = lambda *p: self._nll(np.array(p[:self.deg + 1]), p[-2], p[-1])
        names = [f"c{i}" for i in range(self.deg + 1)] + ["nsig", "mass"]
        m = Minuit(f, *c0, 0.0, mass0, name=names)
        m.errordef = Minuit.LIKELIHOOD
        m.limits["mass"] = (118.0, 132.0)
        if not with_signal:
            m.fixed["nsig"] = True
            m.fixed["mass"] = True
        elif self.fix_mass is not None:
            m.fixed["mass"] = True
        m.migrad()
        m.migrad()
        if with_signal:
            m.hesse()
        return m

    def _fit(self):
        self.m_b = self._minimise(False)
        self.m_sb = self._minimise(True)
        p = self.m_sb.values
        self.nsig, self.nsig_err = p["nsig"], self.m_sb.errors["nsig"]
        self.mass, self.mass_err = p["mass"], self.m_sb.errors["mass"]
        self.coef_sb = np.array([p[f"c{i}"] for i in range(self.deg + 1)])
        q0 = 2 * max(self.m_b.fval - self.m_sb.fval, 0.0)
        self.Z_local = np.sign(self.nsig) * np.sqrt(q0)   # Wilks, 1 dof
        self.p_local = norm.sf(self.Z_local)

    # convenience accessors for plotting / tables
    def background(self, m=CENTRES):
        return _bkg_density(m, self.coef_sb) * BIN_WIDTH

    def total(self, m=CENTRES):
        g = self.nsig * BIN_WIDTH * np.exp(-0.5 * ((m - self.mass) / self.sigma) ** 2) / (
            np.sqrt(2 * np.pi) * self.sigma)
        return self.background(m) + g

    def background_in_window(self, window=WINDOW):
        sel = (CENTRES > window[0]) & (CENTRES < window[1])
        return self.background()[sel].sum()

    def summary(self):
        return {
            "N_sig": self.nsig, "N_sig_err": self.nsig_err,
            "m_H": self.mass, "m_H_err": self.mass_err,
            "Z_local": self.Z_local, "p_local": self.p_local,
            "B_window": self.background_in_window(),
        }


class CategoryFit:
    """Simultaneous fit of several m_yy spectra (e.g. classifier-score
    categories) with ONE common signal strength mu:

        N_sig,c = mu * S_c          (S_c = expected SM Higgs yield from MC)

    Each category keeps its own 4th-order polynomial background. The mass
    is fixed (default 125 GeV); the local significance is from the
    likelihood ratio of mu free vs mu = 0.
    """

    def __init__(self, myy_list, s_expected, sigma, mass=125.0, deg=4):
        self.fits = [MyyFit.__new__(MyyFit) for _ in myy_list]
        for f, m in zip(self.fits, myy_list):
            f.n, _ = np.histogram(m, bins=BINS)
            f.sigma, f.deg, f.fix_mass = sigma, deg, mass
        self.S = np.asarray(s_expected, float)
        self.mass, self.deg = mass, deg
        self._fit()

    def _nll(self, p, mu):
        k = self.deg + 1
        return sum(f._nll(np.asarray(p[i * k:(i + 1) * k]), mu * S, self.mass)
                   for i, (f, S) in enumerate(zip(self.fits, self.S)))

    def _minimise(self, with_signal):
        k, start, names = self.deg + 1, [], []
        for i, f in enumerate(self.fits):
            start += list(np.polyfit(_x(CENTRES), f.n, self.deg,
                                     w=1 / np.sqrt(np.maximum(f.n, 1)))[::-1] / BIN_WIDTH)
            names += [f"c{j}_{i}" for j in range(k)]
        fun = lambda *p: self._nll(p[:-1], p[-1])
        m = Minuit(fun, *start, 1.0 if with_signal else 0.0, name=names + ["mu"])
        m.errordef = Minuit.LIKELIHOOD
        if not with_signal:
            m.fixed["mu"] = True
        m.migrad(); m.migrad()
        if with_signal:
            m.hesse()
        return m

    def _fit(self):
        self.m_b, self.m_sb = self._minimise(False), self._minimise(True)
        self.mu, self.mu_err = self.m_sb.values["mu"], self.m_sb.errors["mu"]
        q0 = 2 * max(self.m_b.fval - self.m_sb.fval, 0.0)
        self.Z_local = np.sign(self.mu) * np.sqrt(q0)
        self.p_local = norm.sf(self.Z_local)

    def expected_Z(self):
        """Asimov expected significance for mu = 1: fit an Asimov dataset made
        of each category's background fit + the SM signal expectation."""
        k = self.deg + 1
        p = self.m_sb.values
        asimov = CategoryFit.__new__(CategoryFit)
        asimov.fits, asimov.S, asimov.mass, asimov.deg = [], self.S, self.mass, self.deg
        for i, f in enumerate(self.fits):
            c = np.array([p[f"c{j}_{i}"] for j in range(k)])
            a = MyyFit.__new__(MyyFit)
            a.sigma, a.deg, a.fix_mass = f.sigma, f.deg, f.fix_mass
            a.n = (np.clip(_bkg_density(CENTRES, c) * BIN_WIDTH, 1e-9, None)
                   + self.S[i] * _gauss_bin_fraction(self.mass, f.sigma))
            asimov.fits.append(a)
        asimov._fit()
        return asimov.Z_local

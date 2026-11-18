# MC weight recipe, class balancing
import numpy as np
import awkward as ak

# Integrated luminosity of each ATLAS Open Data 13 TeV period [fb^-1]
LUMI = {"A": 0.5, "B": 1.9, "C": 2.9, "D": 4.7}


def mc_weight(a, lumi):
    """Per-event MC weight normalising the sample to `lumi` fb^-1.
    XSection is in pb, so x1000 converts pb -> fb."""
    w = (a.mcWeight * a.scaleFactor_PILEUP * a.scaleFactor_PHOTON
         * a.scaleFactor_PhotonTRIGGER * a.XSection * 1000 * lumi / a.SumWeights)
    return ak.to_numpy(w).astype(np.float64)


def balanced(y, w):
    """|weights| with each class normalised to the same total (half the events each).
    MC weights are tiny compared to the unit weights of data events, so without
    this the background would dominate the loss."""
    w = np.abs(np.asarray(w, dtype=float)).copy()
    for c in (0, 1):
        m = y == c
        w[m] *= 0.5 * len(y) / w[m].sum()
    return w

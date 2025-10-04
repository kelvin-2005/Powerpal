# core/calibrator.py
import json
import numpy as np
from typing import Dict, Tuple
from scipy.stats import dirichlet, beta

def fit_temperature(val_preds: np.ndarray, val_targets: np.ndarray,
                    grid=(20, 40, 80, 120, 160, 200)) -> float:
    """
    Pick lambda (Dirichlet concentration scaler) by minimizing average NLL
    over the validation set.
    val_preds, val_targets: shape (n,4), rows sum to 1.
    """
    val_preds = np.clip(val_preds, 1e-9, None)
    val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
    val_targets = np.clip(val_targets, 1e-9, None)
    val_targets = val_targets / val_targets.sum(axis=1, keepdims=True)

    def avg_nll(lmb: float) -> float:
        alphas = np.clip(val_preds * lmb, 1e-6, None)  # (n,4)
        # Row-wise logpdf, then mean
        ll = np.array([dirichlet.logpdf(val_targets[i], alphas[i]) for i in range(len(val_targets))])
        # If any invalids sneak in, mask them
        ll = ll[~np.isnan(ll) & ~np.isinf(ll)]
        return float(-ll.mean())

    best_lmb, best_score = None, float("inf")
    for lmb in grid:
        score = avg_nll(lmb)
        if score < best_score:
            best_lmb, best_score = lmb, score
    return float(best_lmb)

def dirichlet_mean_ci(shares: Dict[str, float], lmb: float, ci: float = 0.95
                      ) -> Dict[str, Tuple[float, float, float]]:
    """
    For a single prediction: return (mean, lo, hi) for each category.
    """
    keys = list(shares.keys())
    s = np.array([max(1e-9, shares[k]) for k in keys], dtype=float)
    s = s / s.sum()
    alpha = np.clip(s * lmb, 1e-6, None)
    alpha0 = float(alpha.sum())
    means = alpha / alpha0

    out = {}
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    for i, k in enumerate(keys):
        a = float(alpha[i])
        b = float(alpha0 - a)
        lo = float(beta.ppf(lo_q, a, b))
        hi = float(beta.ppf(hi_q, a, b))
        out[k] = (float(means[i]), lo, hi)
    return out

def save_lambda(path: str, lmb: float):
    state = {"lambda": float(lmb)}
    with open(path, "w") as f:
        json.dump(state, f)

def load_lambda(path: str, default: float = 120.0) -> float:
    try:
        with open(path, "r") as f:
            return float(json.load(f)["lambda"])
    except Exception:
        return float(default)

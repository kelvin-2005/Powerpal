#!/usr/bin/env python3
"""
Calibrate Dirichlet concentration (lambda) and optional temperature (tau)
from validation predictions/targets.

Features:
- Full Dirichlet negative log-likelihood (with log Beta normalizer).
- Optional temperature scaling p' ∝ p^tau before forming Dirichlet.
- Targets empirical coverage of a chosen CI (default 95%) via a penalty.
- NEW: --val-path lets you choose any val preds CSV (e.g., GBM or Ridge).

Expected CSV columns in --val-path:
  pred_ac, pred_lighting, pred_appliances, pred_other
  true_ac, true_lighting, true_appliances, true_other
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------- Paths (defaults) ---------------- #
BASE = Path(__file__).resolve().parents[1]
DATA_SYN = BASE / "data" / "synthetic"
USER_DIR = BASE / "data" / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VAL_PREDS_PATH = DATA_SYN / "val_preds.csv"  # Ridge default
STATE_PATH = USER_DIR / "model_state.json"

# Must match your model class order
CLASS_ORDER = ["ac", "lighting", "appliances", "other"]


# ---------------- Math utils ---------------- #
def _lgamma_vec(x: np.ndarray) -> np.ndarray:
    v = np.vectorize(math.lgamma, otypes=[float])
    return v(x)


def dirichlet_nll_row(p: np.ndarray, t: np.ndarray, lam: float) -> float:
    """
    Full negative log-likelihood of t under Dirichlet(alpha=lam*p).
    NLL = log B(alpha) - sum((alpha_i - 1) * log t_i)
    where log B(alpha) = sum(lgamma(alpha_i)) - lgamma(sum alpha_i)
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    t = np.clip(t, eps, 1.0); t = t / t.sum()

    alpha = lam * p
    a0 = alpha.sum()

    logB = _lgamma_vec(alpha).sum() - math.lgamma(a0)
    term = float(np.dot((alpha - 1.0), np.log(t)))
    return float(logB - term)


def beta_ci(alpha_i: float, alpha_rest: float, ci: float = 0.95) -> Tuple[float, float]:
    """
    CI for a Dirichlet component marginal: Beta(alpha_i, alpha_rest).
    Uses normal approximation (dependency-free). Adequate for moderate alpha.
    """
    a, b = float(alpha_i), float(alpha_rest)
    tot = a + b
    if tot <= 2.0:
        return (0.0, 1.0)
    mean = a / tot
    var = (a * b) / (tot**2 * (tot + 1.0))
    std = math.sqrt(max(var, 1e-12))
    z = 1.959963984540054  # 97.5% for 95% two-sided
    lo, hi = mean - z * std, mean + z * std
    return (max(0.0, lo), min(1.0, hi))


def empirical_coverage(P: np.ndarray, T: np.ndarray, lam: float, tau: float, ci: float) -> float:
    """Average per-component coverage: fraction of components where t_i is inside Beta CI."""
    eps = 1e-12
    p = np.clip(P, eps, 1.0); p = p / p.sum(axis=1, keepdims=True)
    if abs(tau - 1.0) > 1e-12:
        p = p ** tau
        p = p / p.sum(axis=1, keepdims=True)

    t = np.clip(T, eps, 1.0); t = t / t.sum(axis=1, keepdims=True)

    covered = []
    for i in range(len(p)):
        alpha = lam * p[i]
        a0 = alpha.sum()
        for k in range(p.shape[1]):
            lo, hi = beta_ci(alpha[k], a0 - alpha[k], ci=ci)
            covered.append(1.0 if lo <= t[i, k] <= hi else 0.0)
    return float(np.mean(covered))


@dataclass
class CalibResult:
    lam: float
    tau: float
    nll: float
    coverage: float


def load_val_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    P = df[[f"pred_{c}" for c in CLASS_ORDER]].values.astype(float)
    T = df[[f"true_{c}" for c in CLASS_ORDER]].values.astype(float)
    # safety renorm
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(axis=1, keepdims=True)
    T = np.clip(T, 1e-9, 1.0); T = T / T.sum(axis=1, keepdims=True)
    return P, T


def total_nll(P: np.ndarray, T: np.ndarray, lam: float, tau: float) -> float:
    eps = 1e-12
    p = np.clip(P, eps, 1.0); p = p / p.sum(axis=1, keepdims=True)
    if abs(tau - 1.0) > 1e-12:
        p = p ** tau
        p = p / p.sum(axis=1, keepdims=True)
    nll = 0.0
    for i in range(len(p)):
        nll += dirichlet_nll_row(p[i], T[i], lam)
    return float(nll)


def fit_lambda_tau(P: np.ndarray, T: np.ndarray, target_ci: float = 0.95, fit_tau: bool = True) -> CalibResult:
    """
    Coarse→fine grid search minimizing:
      objective = mean NLL  +  w * (coverage(target_ci) - target_ci)^2
    """
    lam_grid1 = np.geomspace(5, 800, num=12)
    tau_grid1 = np.linspace(0.70, 1.50, num=9) if fit_tau else np.array([1.0])

    best = None
    best_obj = float("inf")
    w_cov = 2000.0  # penalty weight for coverage gap

    for tau in tau_grid1:
        for lam in lam_grid1:
            nll = total_nll(P, T, lam, tau)
            cov = empirical_coverage(P, T, lam, tau, ci=target_ci)
            obj = (nll / len(P)) + w_cov * (cov - target_ci) ** 2
            if obj < best_obj:
                best_obj = obj
                best = CalibResult(lam=lam, tau=tau, nll=nll, coverage=cov)

    # fine search
    lam_lo = max(2.0, best.lam / 2.5)
    lam_hi = best.lam * 2.5
    lam_grid2 = np.geomspace(lam_lo, lam_hi, num=14)

    if fit_tau:
        tau_lo = max(0.55, best.tau - 0.25)
        tau_hi = min(1.70, best.tau + 0.25)
        tau_grid2 = np.linspace(tau_lo, tau_hi, num=11)
    else:
        tau_grid2 = np.array([1.0])

    for tau in tau_grid2:
        for lam in lam_grid2:
            nll = total_nll(P, T, lam, tau)
            cov = empirical_coverage(P, T, lam, tau, ci=target_ci)
            obj = (nll / len(P)) + w_cov * (cov - target_ci) ** 2
            if obj < best_obj:
                best_obj = obj
                best = CalibResult(lam=lam, tau=tau, nll=nll, coverage=cov)

    return best


def main():
    parser = argparse.ArgumentParser(description="Calibrate Dirichlet λ and optional temperature τ.")
    parser.add_argument("--target-ci", type=float, default=0.95, help="Target credible interval (e.g., 0.95)")
    parser.add_argument("--fit-tau", action="store_true", help="Also fit temperature τ (p' ∝ p^τ)")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to val preds CSV (default: data/synthetic/val_preds.csv). "
                             "Use data/synthetic/val_preds_lgbm.csv for GBM.")
    args = parser.parse_args()

    val_path = Path(args.val_path) if args.val_path else DEFAULT_VAL_PREDS_PATH
    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing {val_path}. Run scripts/train_model.py (Ridge) or scripts/train_lgbm.py (GBM) first."
        )

    P, T = load_val_preds(val_path)
    result = fit_lambda_tau(P, T, target_ci=args.target_ci, fit_tau=args.fit_tau)

    state = {
        "dirichlet_lambda": float(result.lam),
        "temperature_tau": float(result.tau),
        "calibration_meta": {
            "target_ci": args.target_ci,
            "achieved_coverage": round(result.coverage, 4),
            "n_val": int(len(P)),
            "val_source": str(val_path)
        }
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

    print(f"[ok] Calibrated λ = {result.lam:.2f}, τ = {result.tau:.3f}")
    print(f"[ok] Empirical coverage @ {args.target_ci:.2f}: {result.coverage:.3f}")
    print(f"[ok] Wrote state → {STATE_PATH}")
    print(f"[info] Using validation file: {val_path}")

if __name__ == "__main__":
    main()

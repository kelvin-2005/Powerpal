#!/usr/bin/env python3
"""
Generate synthetic household-month data for training the SoftmaxShareModel.

Outputs (by default):
  data/synthetic/home_train.csv
  data/synthetic/home_val.csv

Columns:
  bill_aed, tariff, home_type, size, occupants, setpoint, led_pct,
  ac_share, lighting_share, appliances_share, other_share

How it works (high level):
- Sample realistic household features (UAE-style seasonality).
- Compute physics-informed mean energy shares (AC ↑ in summer / low setpoint;
  Lighting ↓ with high LED%; Villas & larger size tilt toward AC/Lighting).
- Sample final shares from a Dirichlet distribution centered on those means.
- Compute total kWh and bill_aed = kWh * tariff (with a little noise).

Usage:
  python scripts/generate_synthetic.py --n-train 10000 --n-val 2000 --seed 1337
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ----------------------------- Defaults ----------------------------- #
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reasonable tariff ranges (AED/kWh) seen across UAE utilities
TARIFF_RANGE = (0.25, 0.50)  # inclusive-ish

# Occupants distribution (lightly skewed to 2–4)
OCCUPANT_CHOICES = np.array([1,2,3,4,5,6,7,8])
OCCUPANT_PROBS   = np.array([0.05,0.15,0.25,0.23,0.15,0.10,0.05,0.02])
OCCUPANT_PROBS  /= OCCUPANT_PROBS.sum()

# Home type distribution
HOME_TYPES = np.array(["apartment", "villa"])
HOME_TYPE_PROBS = np.array([0.65, 0.35])

# Size distribution
SIZES = np.array(["S", "M", "L"])
SIZE_PROBS = np.array([0.35, 0.45, 0.20])

# LED share distribution (bimodal-ish around 30–60)
def sample_led_pct(n: int, rng: np.random.Generator) -> np.ndarray:
    # Mix of two betas (skewed low + mid)
    mix = rng.random(n)
    a1, b1 = 2.0, 3.5  # lower LEDs
    a2, b2 = 4.0, 3.0  # mid LEDs
    base = np.where(mix < 0.5,
                    rng.beta(a1, b1, size=n),
                    rng.beta(a2, b2, size=n))
    return np.clip((base * 100).round(0), 0, 100)

# Setpoint distribution (cooler in hot months)
def sample_setpoint(months: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    setpt = np.empty_like(months, dtype=float)
    hot_months = np.isin(months, [5,6,7,8,9,10])
    # Hot months: lower setpoint (cooler), e.g., mean 22.5 ± 1.5
    setpt[hot_months] = rng.normal(loc=22.5, scale=1.5, size=hot_months.sum())
    # Cooler months: slightly higher setpoint, e.g., 23.5 ± 1.2
    setpt[~hot_months] = rng.normal(loc=23.5, scale=1.2, size=(~hot_months).sum())
    return np.clip(setpt.round(0), 18, 26)

# Tariff per row
def sample_tariff(n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = TARIFF_RANGE
    return rng.uniform(lo, hi, size=n).round(2)

# Base monthly kWh scaling (rough)
def base_kwh_scale(home_type: np.ndarray, size: np.ndarray, occupants: np.ndarray) -> np.ndarray:
    """
    Returns a multiplicative factor ~ [0.6 .. 2.2] to scale a base 400 kWh/month.
    Villas & larger size → higher; more occupants → higher.
    """
    ht_mult = np.where(home_type == "villa", 1.25, 0.95)
    size_mult = np.where(size == "L", 1.35, np.where(size == "M", 1.10, 0.90))
    occ_mult = 0.85 + 0.12 * np.clip(occupants - 1, 0, 6)  # +12% per extra occupant
    return ht_mult * size_mult * occ_mult

def seasonal_kwh_bump(months: np.ndarray) -> np.ndarray:
    """
    Increase kWh in hot months (May–Oct).
    """
    hot = np.isin(months, [5,6,7,8,9,10])
    return np.where(hot, 1.25, 0.90)  # hot months +25%, cooler months -10%

# ---------------- Physics-informed mean share model ---------------- #
def mean_shares(
    home_type: str,
    size: str,
    occupants: int,
    setpoint_c: float,
    led_pct: float,
    month: int
) -> Tuple[float, float, float, float]:
    """
    Produce mean shares for (ac, lighting, appliances, other) that sum to 1.
    - AC ↑ for villas, larger size, lower setpoint, hot months.
    - Lighting ↓ with higher LED%.
    - Appliances ↑ with occupants.
    - 'Other' soaks residual, floored to a small minimum.
    """

    # Start with neutral baseline
    ac      = 0.40
    light   = 0.18
    apps    = 0.28
    other   = 0.14

    # Home type & size effects (AC & Lighting slightly higher in bigger villas)
    if home_type == "villa":
        ac   += 0.05
        light += 0.01
    if size == "M":
        ac   += 0.02
        apps += 0.01
    elif size == "L":
        ac   += 0.04
        apps += 0.02

    # Seasonal effect (hot months → more AC)
    if month in [5,6,7,8,9,10]:
        ac += 0.06

    # Setpoint sensitivity (~4% AC per +1°C away from 22°C baseline)
    # Lower setpoint (e.g., 20) → higher AC share.
    ac += -0.04 * (setpoint_c - 22.0)

    # LED effect: each +10% LED reduces lighting ~8% from baseline share
    light *= (1.0 - 0.08 * (led_pct / 10.0))

    # Occupants effect on appliances (each +1 occupant → +3% on appliances share)
    apps *= (1.0 + 0.03 * max(0, occupants - 1))

    # Clamp nonnegatives
    ac    = max(ac, 0.02)
    light = max(light, 0.02)
    apps  = max(apps, 0.05)

    # Renormalize prelim and compute 'other'
    prelim = np.array([ac, light, apps])
    prelim = prelim / prelim.sum()  # scale to 1 among first 3
    # Reserve some mass for 'other' (~10–20%) based on setpoint/LED (more efficiency → more "other")
    other = 0.12 + 0.03 * ((setpoint_c - 22.0) / 4.0) + 0.03 * (led_pct / 100.0)
    other = float(np.clip(other, 0.08, 0.22))

    # Now scale back so total = 1 with 'other'
    scale = (1.0 - other) / prelim.sum()
    ac, light, apps = (prelim * scale).tolist()

    # Final tidy normalize to avoid drift
    arr = np.array([ac, light, apps, other], dtype=float)
    arr = np.clip(arr, 1e-4, 1.0)
    arr = arr / arr.sum()
    return tuple(arr.tolist())


def dirichlet_sample(mean: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from Dirichlet with given mean and concentration kappa.
    mean: 1D array summing to 1
    kappa: scalar concentration (> 0). Higher = lower variance.
    """
    alpha = np.clip(mean, 1e-6, None) * float(kappa)
    return rng.dirichlet(alpha)


# ------------------------------ Main gen ------------------------------ #
def synthesize(n: int, rng: np.random.Generator) -> pd.DataFrame:
    # Sample months (spread across year for seasonality)
    months = rng.integers(1, 13, size=n, endpoint=True)

    # Sample household attributes
    home_type = rng.choice(HOME_TYPES, size=n, p=HOME_TYPE_PROBS)
    size_vals = rng.choice(SIZES, size=n, p=SIZE_PROBS)
    occupants = rng.choice(OCCUPANT_CHOICES, size=n, p=OCCUPANT_PROBS)
    led_vals  = sample_led_pct(n, rng)
    setpts    = sample_setpoint(months, rng)
    tariff    = sample_tariff(n, rng)

    # Compute mean shares & sample final shares
    ac_s = np.empty(n, dtype=float)
    li_s = np.empty(n, dtype=float)
    ap_s = np.empty(n, dtype=float)
    ot_s = np.empty(n, dtype=float)

    for i in range(n):
        mu = np.array(mean_shares(
            home_type=home_type[i],
            size=size_vals[i],
            occupants=int(occupants[i]),
            setpoint_c=float(setpts[i]),
            led_pct=float(led_vals[i]),
            month=int(months[i])
        ), dtype=float)

        # Concentration ~ [80..200], slightly higher for apartments (less variance)
        base_kappa = 80.0 if home_type[i] == "villa" else 110.0
        kappa = base_kappa + rng.uniform(0, 90)
        sample = dirichlet_sample(mu, kappa, rng)

        ac_s[i], li_s[i], ap_s[i], ot_s[i] = sample

    # Build kWh and bills
    base = 400.0  # base kWh/month neutral
    scale = base_kwh_scale(home_type, size_vals, occupants) * seasonal_kwh_bump(months)
    kwh_total = base * scale
    # Add a bit of noise to kWh totals (±7%)
    kwh_total *= (1.0 + rng.normal(loc=0.0, scale=0.07, size=n))
    kwh_total = np.clip(kwh_total, 120.0, 4000.0)

    bill_aed = kwh_total * tariff
    # Add small billing noise (rounding, fees), ±3%
    bill_aed *= (1.0 + rng.normal(0.0, 0.03, size=n))
    bill_aed = np.clip(bill_aed, 50.0, 5000.0)

    df = pd.DataFrame({
        "bill_aed": bill_aed.round(0),
        "tariff": tariff.round(2),
        "home_type": home_type,
        "size": size_vals,
        "occupants": occupants.astype(int),
        "setpoint": setpts.astype(int),
        "led_pct": led_vals.astype(int),
        "ac_share": ac_s,
        "lighting_share": li_s,
        "appliances_share": ap_s,
        "other_share": ot_s,
    })

    # Final sanity: ensure rows sum to ~1.0
    shares_sum = df[["ac_share","lighting_share","appliances_share","other_share"]].sum(axis=1)
    df.loc[:, ["ac_share","lighting_share","appliances_share","other_share"]] = (
        df[["ac_share","lighting_share","appliances_share","other_share"]].div(shares_sum, axis=0)
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic household-month data.")
    parser.add_argument("--n-train", type=int, default=10000, help="Number of training rows")
    parser.add_argument("--n-val", type=int, default=2000, help="Number of validation rows")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: data/synthetic)"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Generating synthetic data → {outdir}")
    print(f"       n_train={args.n_train}, n_val={args.n_val}, seed={args.seed}")

    df_train = synthesize(args.n_train, rng)
    df_val   = synthesize(args.n_val,   rng)

    # Light QA prints
    for name, df in [("train", df_train), ("val", df_val)]:
        sums = df[["ac_share","lighting_share","appliances_share","other_share"]].sum(axis=1)
        print(f"[info] {name}: share sum mean={sums.mean():.6f}, std={sums.std():.6f}")
        print(f"[info] {name}: bill AED mean={df['bill_aed'].mean():.1f}, tariff mean={df['tariff'].mean():.2f}")

    # Save
    train_path = outdir / "home_train.csv"
    val_path   = outdir / "home_val.csv"
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"[ok] Wrote: {train_path}  ({len(df_train)} rows)")
    print(f"[ok] Wrote: {val_path}    ({len(df_val)} rows)")
    print("[done] Synthetic data generation complete.")


if __name__ == "__main__":
    main()

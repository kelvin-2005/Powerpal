# core/data_gen.py
import numpy as np
import pandas as pd
from typing import Tuple
from .config import BASE_SHARES, RULES, HOME_TYPES, SIZES

def _renorm_shares(d: dict) -> dict:
    # clamp to >= 0 and renormalize to sum=1
    keys = list(d.keys())
    arr = np.clip(np.array([d[k] for k in keys]), 0, None)
    s = arr.sum()
    if s <= 0:
        arr = np.array([1.0, 0.0, 0.0, 0.0])
        s = 1.0
    arr = arr / s
    return {k: v for k, v in zip(keys, arr)}

def _apply_rules(row) -> dict:
    shares = dict(BASE_SHARES)

    # Villa & Large effects on AC
    if row["home_type"] == "villa":
        shares["ac"] += RULES["villa_ac_pp"]
        shares["appliances"] -= RULES["villa_ac_pp"]/2
        shares["water"] -= RULES["villa_ac_pp"]/2

    if row["size"] == "L":
        shares["ac"] += RULES["large_ac_pp"]
        shares["appliances"] -= RULES["large_ac_pp"]/2
        shares["lighting"] -= RULES["large_ac_pp"]/2

    # Occupants effects on apps + water
    if row["occupants"] >= 4:
        shares["appliances"] += RULES["occupants_apps_pp"]
        shares["water"] += RULES["occupants_water_pp"]
        shares["ac"] -= (RULES["occupants_apps_pp"] + RULES["occupants_water_pp"]) * 0.7
        shares["lighting"] -= (RULES["occupants_apps_pp"] + RULES["occupants_water_pp"]) * 0.3

    # LED impact in generator (reduce lighting share if LED high)
    led_frac = row["led_pct"] / 100.0
    lighting = shares["lighting"] * (1.0 - led_frac*RULES["led_reduction_factor"])
    # take the reduced amount and redistribute proportionally to others
    reduction = shares["lighting"] - lighting
    shares["lighting"] = lighting
    redist_keys = ["ac", "appliances", "water"]
    for k in redist_keys:
        shares[k] += reduction * (BASE_SHARES[k] / (BASE_SHARES["ac"] + BASE_SHARES["appliances"] + BASE_SHARES["water"]))

    # Small Gaussian noise then renormalize
    for k in shares:
        shares[k] = max(0.0, shares[k] + np.random.normal(0, 0.01))
    shares = _renorm_shares(shares)
    return shares

def make_synthetic(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    home_type = rng.choice(HOME_TYPES, size=n, p=[0.7, 0.3])  # more apartments
    size = rng.choice(SIZES, size=n, p=[0.4, 0.45, 0.15])     # few large
    occupants = rng.integers(1, 6, size=n)
    setpoint = rng.integers(21, 27, size=n)                   # 21..26°C typical
    led_pct = rng.integers(0, 101, size=n)                    # 0..100%
    tariff = rng.uniform(0.20, 0.40, size=n)                  # AED/kWh
    bill_aed = rng.uniform(150, 1200, size=n)

    df = pd.DataFrame({
        "home_type": home_type,
        "size": size,
        "occupants": occupants,
        "setpoint": setpoint,
        "led_pct": led_pct,
        "tariff": tariff,
        "bill_aed": bill_aed,
    })

    shares_list = []
    for _, row in df.iterrows():
        shares = _apply_rules(row)
        shares_list.append(shares)
    shares_df = pd.DataFrame(shares_list)
    df = pd.concat([df, shares_df.add_prefix("true_")], axis=1)

    # derived kWh labels
    kwh = df["bill_aed"] / df["tariff"]
    for k in ["ac", "appliances", "lighting", "water"]:
        df[f"true_kwh_{k}"] = kwh * df[f"true_{k}"]

    return df

def train_val_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed)  # shuffle
    n_val = int(len(df) * val_frac)
    val = df.iloc[:n_val].reset_index(drop=True)
    train = df.iloc[n_val:].reset_index(drop=True)
    return train, val

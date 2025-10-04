# core/online_learner.py
import numpy as np
import pandas as pd
from typing import Dict

def fit_or_update(df_history: pd.DataFrame) -> Dict[str, float]:
    """
    df_history columns: month,bill_aed,tariff,home_type,size,occupants,setpoint,led_pct
    Very small linear fit to learn two sensitivities:
    - beta_ac_per_c: effect of setpoint on (implied) AC energy
    - led_efficacy: effect of LED% on lighting energy

    This is a toy learner; if no data, return conservative defaults.
    """
    if df_history is None or len(df_history) < 2:
        return {"beta_ac_per_c": 0.04, "led_efficacy": 0.70}

    # Construct rough targets from differences month-to-month
    df = df_history.copy().sort_values("month")
    df["kwh"] = df["bill_aed"] / df["tariff"]
    # finite diffs
    df["dkwh"] = df["kwh"].diff()
    df["dset"] = df["setpoint"].diff()
    df["dled"] = (df["led_pct"].diff() / 100.0)

    # Simple robust estimates ignoring NaNs
    beta_ac = 0.04
    led_eff = 0.70
    try:
        valid = df.dropna()
        if len(valid) >= 2:
            # Linear regress dkwh on dset and dled with tiny ridge
            X = valid[["dset", "dled"]].values
            y = -valid["dkwh"].values  # assume raising setpoint reduces kWh (negative dkwh -> positive savings)
            XtX = X.T @ X + 1e-3*np.eye(2)
            Xty = X.T @ y
            coef = np.linalg.solve(XtX, Xty)
            beta_ac = max(0.01, min(0.08, coef[0]))  # clamp
            led_eff = max(0.3, min(0.85, coef[1]))
    except Exception:
        pass

    return {"beta_ac_per_c": float(beta_ac), "led_efficacy": float(led_eff)}

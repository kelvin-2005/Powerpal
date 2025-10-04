# core/features.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Profile:
    bill_aed: float
    tariff: float
    home_type: str          # "apartment" | "villa"
    size: str               # "S" | "M" | "L"
    occupants: int
    setpoint: int
    led_pct: int

def _one_hot(value: str, choices: list[str]) -> list[float]:
    return [1.0 if value == c else 0.0 for c in choices]

def build_feature_vector(p: Profile, extra: dict | None = None) -> np.ndarray:
    """
    Base engineered features + optional extras (month one-hot, cdd_proxy).
    """
    # base
    ht = _one_hot(p.home_type, ["apartment","villa"])
    sz = _one_hot(p.size, ["S","M","L"])
    x = [
        float(p.bill_aed),
        float(p.tariff),
        float(p.occupants),
        float(p.setpoint),
        float(p.led_pct),
        # simple interactions
        float(p.bill_aed) / max(p.tariff, 1e-6),   # implied kWh
        float(p.occupants) * 1.0,
        float(p.setpoint) * 1.0,
        float(p.led_pct) / 100.0,
    ] + ht + sz

    # extras: month one-hot (1..12) + cdd proxy
    if extra:
        m = int(extra.get("month", 0))
        month_oh = [1.0 if i == m else 0.0 for i in range(1, 13)]
        cdd = float(extra.get("cdd_proxy", 0.0))
        x.extend(month_oh)
        x.append(cdd)

    return np.array(x, dtype=float)

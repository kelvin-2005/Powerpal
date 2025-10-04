# core/savings.py
from typing import Dict, Tuple
from .config import DEFAULT_AC_ELASTICITY_PER_C, DEFAULT_LED_EFFICACY

def bill_to_kwh(bill_aed: float, tariff: float) -> float:
    return max(0.0, bill_aed / max(1e-6, tariff))

def ac_savings(kwh_ac: float, delta_c: float, elasticity: float = DEFAULT_AC_ELASTICITY_PER_C) -> float:
    """
    Positive delta_c means raising setpoint.
    Savings = kwh_ac * elasticity * delta_c
    """
    if delta_c <= 0:
        return 0.0
    return max(0.0, kwh_ac * elasticity * delta_c)

def led_savings(kwh_lighting: float, led_pct_now: float, target_led_pct: float = 100.0, efficacy: float = DEFAULT_LED_EFFICACY) -> float:
    """
    Save efficacy fraction on the portion that is currently NOT LED.
    """
    led_now = max(0.0, min(100.0, led_pct_now)) / 100.0
    led_target = max(led_now, min(1.0, target_led_pct / 100.0))
    non_led_fraction_replaced = max(0.0, led_target - led_now)
    return max(0.0, kwh_lighting * non_led_fraction_replaced * efficacy)

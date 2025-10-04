# core/config.py
from dataclasses import dataclass

# ---------- Defaults ----------
DEFAULT_TARIFF_AED_PER_KWH = 0.30   # editable in UI
CO2_KG_PER_KWH = 0.42               # rough grid factor; adjust if you want

# Priors for shares (must sum to 1.0)
BASE_SHARES = {
    "ac": 0.50,
    "appliances": 0.25,
    "lighting": 0.10,
    "water": 0.15,
}

# Rule weights that shape synthetic data + initial adjustments
RULES = {
    "villa_ac_pp": 0.10,     # +10 percentage points to AC if villa
    "large_ac_pp": 0.05,     # +5pp AC if large area
    "occupants_apps_pp": 0.05,  # +5pp apps if many occupants
    "occupants_water_pp": 0.05, # +5pp water if many occupants
    "led_reduction_factor": 0.50,  # LEDs halve lighting demand in generator
}

# “Savings engines” default elasticities (overridden by online learner if present)
DEFAULT_AC_ELASTICITY_PER_C = 0.04     # ~4% per +1°C
DEFAULT_LED_EFFICACY = 0.70            # 70% saving on non-LED lighting share

# Dirichlet calibration temperature (overwritten by calibrator script)
DEFAULT_DIRICHLET_LAMBDA = 120.0

HOME_TYPES = ["apartment", "villa"]
SIZES = ["S", "M", "L"]

@dataclass
class UIBounds:
    min_bill: float = 50.0
    max_bill: float = 3000.0
    min_tariff: float = 0.10
    max_tariff: float = 1.00
    min_occupants: int = 1
    max_occupants: int = 6
    min_setpoint: int = 20
    max_setpoint: int = 28
    min_led_pct: int = 0
    max_led_pct: int = 100

BOUNDS = UIBounds()

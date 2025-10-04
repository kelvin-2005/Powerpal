# scripts/demo_seed_user.py
"""
Seeds the app with a few demo entries so the Dashboard isn't empty the first time.
- Creates data/user/history.csv with 2 recent months
- Creates/updates data/user/actions.csv with 2 committed actions (today)
Run from the project root:
  python -m scripts.demo_seed_user
Options:
  --force   overwrite existing history/actions instead of appending
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

BASE = Path(__file__).resolve().parents[1]
USER_DIR = BASE / "data" / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS_PATH = USER_DIR / "actions.csv"
HISTORY_PATH = USER_DIR / "history.csv"

def seed_history(force: bool = False):
    today = datetime.now()
    last_month = (today.replace(day=1) - pd.offsets.MonthBegin(1)).to_pydatetime()
    prev_month = (last_month.replace(day=1) - pd.offsets.MonthBegin(1)).to_pydatetime()

    hist_rows = [
        {
            "month": prev_month.strftime("%Y-%m"),
            "bill_aed": 380.0,
            "tariff": 0.30,
            "home_type": "apartment",
            "size": "M",
            "occupants": 3,
            "setpoint": 24,
            "led_pct": 40,
        },
        {
            "month": last_month.strftime("%Y-%m"),
            "bill_aed": 410.0,
            "tariff": 0.30,
            "home_type": "apartment",
            "size": "M",
            "occupants": 3,
            "setpoint": 24,
            "led_pct": 50,
        },
    ]
    df = pd.DataFrame(hist_rows)

    if HISTORY_PATH.exists() and not force:
        # Append only months we don't already have
        old = pd.read_csv(HISTORY_PATH)
        keycols = ["month"]
        merged = pd.concat([old, df]).drop_duplicates(subset=keycols, keep="first")
        merged.to_csv(HISTORY_PATH, index=False)
        print(f"Appended history -> {HISTORY_PATH}")
    else:
        df.to_csv(HISTORY_PATH, index=False)
        print(f"Wrote history -> {HISTORY_PATH}")

def seed_actions(force: bool = False):
    today = datetime.now().strftime("%Y-%m-%d")
    # Two believable actions with plausible savings
    rows = [
        {"date": today, "action_type": "setpoint", "delta": 1,   "est_kwh_saved": 45.0, "est_aed_saved": 14.0, "source": "seed"},
        {"date": today, "action_type": "led",      "delta": 50,  "est_kwh_saved": 22.0, "est_aed_saved": 6.6,  "source": "seed"},
    ]
    df = pd.DataFrame(rows)
    if ACTIONS_PATH.exists() and not force:
        old = pd.read_csv(ACTIONS_PATH)
        # Avoid duplicate “today+seed” rows
        merged = pd.concat([old, df]).drop_duplicates(subset=["date", "action_type", "source"], keep="first")
        merged.to_csv(ACTIONS_PATH, index=False)
        print(f"Appended actions -> {ACTIONS_PATH}")
    else:
        df.to_csv(ACTIONS_PATH, index=False)
        print(f"Wrote actions -> {ACTIONS_PATH}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()
    seed_history(force=args.force)
    seed_actions(force=args.force)
    print("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# path shim
import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from core.features import Profile, build_feature_vector
from core.model import SoftmaxShareModel, CLASS_ORDER

BASE = Path(__file__).resolve().parents[1]
DATA_SYN = BASE / "data" / "synthetic"; DATA_SYN.mkdir(parents=True, exist_ok=True)
TRAIN_CSV = DATA_SYN / "home_train.csv"
VAL_CSV   = DATA_SYN / "home_val.csv"
MODEL_PATH     = DATA_SYN / "softmax_model.json"
VAL_PREDS_PATH = DATA_SYN / "val_preds.csv"

TARGET_COLS = {"ac":"ac_share","lighting":"lighting_share","appliances":"appliances_share","other":"other_share"}

def _row_to_x(row):
    p = Profile(
        bill_aed=float(row["bill_aed"]),
        tariff=float(row["tariff"]),
        home_type=str(row["home_type"]),
        size=str(row["size"]),
        occupants=int(row["occupants"]),
        setpoint=int(row["setpoint"]),
        led_pct=int(row["led_pct"]),
    )
    extra = {
        "month": int(row.get("month", 0)),
        "cdd_proxy": float(row.get("cdd_proxy", 0.0)),
    }
    return build_feature_vector(p, extra=extra)

def _build_XY(df: pd.DataFrame):
    X = np.vstack([_row_to_x(r) for _, r in df.iterrows()])
    y_cols = [TARGET_COLS[c] for c in CLASS_ORDER]
    Y = df[y_cols].values.astype(float)
    Y = Y / np.clip(Y.sum(axis=1, keepdims=True), 1e-9, None)
    return X, Y

def main():
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)

    X_train, Y_train = _build_XY(train)
    X_val,   Y_val   = _build_XY(val)

    model = SoftmaxShareModel()
    model.train(X_train, Y_train)
    model.save(str(MODEL_PATH))
    print(f"[ok] Saved model → {MODEL_PATH}")

    preds = np.vstack([list(model.predict_shares(X_val[i]).values()) for i in range(len(X_val))])
    preds = np.clip(preds, 1e-9, 1.0); preds = preds / preds.sum(axis=1, keepdims=True)

    mae_pp = (np.abs(preds - Y_val).mean(axis=0) * 100.0)
    print("[info] Validation MAE (pp):", {c: round(mae_pp[i],2) for i,c in enumerate(CLASS_ORDER)}, "Avg:", round(mae_pp.mean(),2))

    out = pd.DataFrame(preds, columns=[f"pred_{c}" for c in CLASS_ORDER])
    for i, c in enumerate(CLASS_ORDER):
        out[f"true_{c}"] = Y_val[:, i]
    out.to_csv(VAL_PREDS_PATH, index=False)
    print(f"[ok] Saved validation predictions → {VAL_PREDS_PATH}")

if __name__ == "__main__":
    main()

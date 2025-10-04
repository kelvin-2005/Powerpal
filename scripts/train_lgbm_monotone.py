#!/usr/bin/env python3
# Train LightGBM (centered logits) with monotonic constraints + small ensemble

import sys, os, json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.features import Profile, build_feature_vector
from core.model import CLASS_ORDER  # ["ac","appliances","lighting","other"] in your repo

BASE = Path(__file__).resolve().parents[1]
DATA_SYN = BASE / "data" / "synthetic"
DATA_SYN.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_SYN / "home_train.csv"
VAL   = DATA_SYN / "home_val.csv"

OUT_MODEL = DATA_SYN / "softmax_model_lgbm.json"          # app already prefers this
OUT_VAL   = DATA_SYN / "val_preds_lgbm.csv"               # optional

TARGET_COLS = {"ac":"ac_share","lighting":"lighting_share","appliances":"appliances_share","other":"other_share"}

# --- Feature order in build_feature_vector (must match your core/features.py) ---
# 0  bill_aed
# 1  tariff
# 2  occupants (copy 1)
# 3  setpoint  (copy 1)
# 4  led_pct
# 5  implied_kwh = bill/tariff
# 6  occupants (copy 2)
# 7  setpoint  (copy 2)
# 8  led_pct/100
# 9..10  home_type one-hot ["apartment","villa"]
# 11..13 size one-hot ["S","M","L"]
# 14..25 month one-hot (1..12)
# 26 cdd_proxy

def row_to_x(row):
    p = Profile(
        bill_aed=float(row["bill_aed"]), tariff=float(row["tariff"]),
        home_type=str(row["home_type"]), size=str(row["size"]),
        occupants=int(row["occupants"]), setpoint=int(row["setpoint"]),
        led_pct=int(row["led_pct"]),
    )
    extra = {"month": int(row.get("month", 0)), "cdd_proxy": float(row.get("cdd_proxy", 0.0))}
    return build_feature_vector(p, extra=extra)

def to_centered_logits(Y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    log_s = np.log(np.clip(Y, eps, 1.0))
    return log_s - log_s.mean(axis=1, keepdims=True)

def probs_from_logits(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.mean(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    return P

def build_monotone_constraints(target: str, n_features: int) -> list[int]:
    """Return per-feature monotone constraints for a given target.
       +1: increasing, -1: decreasing, 0: no constraint.
    """
    m = [0] * n_features
    # indices for convenience
    OCC1, OCC2 = 2, 6
    SET1, SET2 = 3, 7
    LED1, LEDN = 4, 8
    CDD = 26

    if target == "ac":
        # AC share should DECREASE with setpoint; INCREASE with heat (cdd_proxy)
        m[SET1] = -1; m[SET2] = -1
        m[CDD]  = +1
        # Leave others unconstrained (0) to avoid over-constraining.
    elif target == "lighting":
        # Lighting share should DECREASE as LED% rises; slight INCREASE with occupants
        m[LED1] = -1; m[LEDN] = -1
        m[OCC1] = +1; m[OCC2] = +1
    elif target == "appliances":
        # Appliances share tends to INCREASE with occupants
        m[OCC1] = +1; m[OCC2] = +1
    else:
        # "other": leave unconstrained
        pass
    return m

def main():
    train = pd.read_csv(TRAIN)
    val   = pd.read_csv(VAL)

    X_train = np.vstack([row_to_x(r) for _, r in train.iterrows()])
    X_val   = np.vstack([row_to_x(r) for _, r in val.iterrows()])

    y_cols = [TARGET_COLS[c] for c in CLASS_ORDER]
    Y_train = train[y_cols].values.astype(float)
    Y_val   = val[y_cols].values.astype(float)

    # scale like baseline
    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0) + 1e-8
    Xs_train = (X_train - feat_mean) / feat_std
    Xs_val   = (X_val   - feat_mean) / feat_std

    T_train = to_centered_logits(Y_train)
    T_val   = to_centered_logits(Y_val)

    n_features = Xs_train.shape[1]
    seeds = [42, 7, 99]   # small ensemble

    # Train per target with monotonic constraints + ensembling
    models = {c: [] for c in CLASS_ORDER}
    best_iters = {c: [] for c in CLASS_ORDER}

    for i, c in enumerate(CLASS_ORDER):
        mono = build_monotone_constraints(c, n_features)

        for seed in seeds:
            dtrain = lgb.Dataset(Xs_train, label=T_train[:, i])
            dval   = lgb.Dataset(Xs_val,   label=T_val[:, i], reference=dtrain)

            params = dict(
                objective="regression",
                metric="l2",
                learning_rate=0.04,
                num_leaves=31,
                min_data_in_leaf=60,
                feature_fraction=0.85,
                bagging_fraction=0.9,
                bagging_freq=1,
                monotone_constraints="(" + ",".join(str(v) for v in mono) + ")",
                verbosity=-1,
                seed=seed,
            )
            callbacks = [
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(0)
            ]
            booster = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=1800,
                callbacks=callbacks
            )
            models[c].append(booster)
            best_iters[c].append(booster.best_iteration)

    # Validate: average ensemble predictions per class
    Z_val_pred = []
    for i, c in enumerate(CLASS_ORDER):
        preds = np.column_stack([
            m.predict(Xs_val, num_iteration=best_iters[c][k]) for k, m in enumerate(models[c])
        ])
        Z_val_pred.append(preds.mean(axis=1))
    Z_val_pred = np.column_stack(Z_val_pred)
    P_val_pred = probs_from_logits(Z_val_pred)

    mae_pp = (np.abs(P_val_pred - Y_val).mean(axis=0) * 100.0)
    print("[info] Monotone-GBM Ens. Val MAE (pp):",
          {c: round(float(mae_pp[i]),2) for i,c in enumerate(CLASS_ORDER)},
          "Avg:", round(float(mae_pp.mean()),2))

    # Save bundle (ensemble)
    bundle = {
        "type": "lgbm_centered_logits_ensemble",
        "class_order": CLASS_ORDER,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "best_iterations": {c: [int(x) for x in best_iters[c]] for c in CLASS_ORDER},
        "models_ensemble": {c: [m.model_to_string() for m in models[c]] for c in CLASS_ORDER},
        "seeds": seeds,
    }
    with open(OUT_MODEL, "w") as f:
        json.dump(bundle, f)
    print(f"[ok] Saved monotone-ensemble GBM → {OUT_MODEL}")

    # Optional: write val preds for calibration
    out = pd.DataFrame(P_val_pred, columns=[f"pred_{c}" for c in CLASS_ORDER])
    for i, c in enumerate(CLASS_ORDER):
        out[f"true_{c}"] = Y_val[:, i]
    out.to_csv(OUT_VAL, index=False)
    print(f"[ok] Saved validation predictions → {OUT_VAL}")

if __name__ == "__main__":
    main()

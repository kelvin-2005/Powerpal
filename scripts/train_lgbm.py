#!/usr/bin/env python3
# Train LightGBM regressors on centered logits (drop-in for Ridge baseline)

import sys, os, json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.features import Profile, build_feature_vector
from core.model import CLASS_ORDER  # reuse category order from baseline

BASE = Path(__file__).resolve().parents[1]
DATA_SYN = BASE / "data" / "synthetic"
DATA_SYN.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_SYN / "home_train.csv"
VAL   = DATA_SYN / "home_val.csv"

# output file (app will prefer this if present)
GBM_MODEL_PATH = DATA_SYN / "softmax_model_lgbm.json"
VAL_PREDS_PATH = DATA_SYN / "val_preds_lgbm.csv"  # optional, if you want alternate calibration

TARGET_COLS = {"ac":"ac_share","lighting":"lighting_share","appliances":"appliances_share","other":"other_share"}

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

def from_logits_to_probs(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.mean(axis=1, keepdims=True)  # safety center
    P = np.exp(Z)
    P = P / P.sum(axis=1, keepdims=True)
    return P

def main():
    train = pd.read_csv(TRAIN)
    val   = pd.read_csv(VAL)

    X_train = np.vstack([row_to_x(r) for _, r in train.iterrows()])
    X_val   = np.vstack([row_to_x(r) for _, r in val.iterrows()])

    y_cols = [TARGET_COLS[c] for c in CLASS_ORDER]
    Y_train = train[y_cols].values.astype(float)
    Y_val   = val[y_cols].values.astype(float)

    # scaler (match baseline behavior)
    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0) + 1e-8
    Xs_train = (X_train - feat_mean) / feat_std
    Xs_val   = (X_val   - feat_mean) / feat_std

    T_train = to_centered_logits(Y_train)
    T_val   = to_centered_logits(Y_val)

    models = {}
    best_iters = {}
    for i, c in enumerate(CLASS_ORDER):
        dtrain = lgb.Dataset(Xs_train, label=T_train[:, i])
        dval   = lgb.Dataset(Xs_val,   label=T_val[:, i], reference=dtrain)

        params = dict(
            objective="regression",
            metric="l2",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=60,
            feature_fraction=0.8,
            bagging_fraction=0.9,
            bagging_freq=1,
            verbosity=-1,
            seed=42,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=120, verbose=False),
            lgb.log_evaluation(0)
        ]

        booster = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=1500,
            callbacks=callbacks
        )
        models[c] = booster
        best_iters[c] = booster.best_iteration


    # Evaluate on val
    Z_val_pred = np.column_stack([models[c].predict(Xs_val, num_iteration=best_iters[c]) for c in CLASS_ORDER])
    P_val_pred = from_logits_to_probs(Z_val_pred)
    mae_pp = (np.abs(P_val_pred - Y_val).mean(axis=0) * 100.0)
    print("[info] LGBM Validation MAE (pp):", {c: round(float(mae_pp[i]),2) for i,c in enumerate(CLASS_ORDER)},
          "Avg:", round(float(mae_pp.mean()),2))

    # Save a JSON bundle the app can load
    bundle = {
        "type": "lgbm_centered_logits",
        "class_order": CLASS_ORDER,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "best_iterations": {c: int(best_iters[c]) for c in CLASS_ORDER},
        "models": {c: models[c].model_to_string() for c in CLASS_ORDER},  # text dump
    }
    with open(GBM_MODEL_PATH, "w") as f:
        json.dump(bundle, f)
    print(f"[ok] Saved GBM model → {GBM_MODEL_PATH}")

    # Optional: save val preds/targets if you want separate calibration
    out = pd.DataFrame(P_val_pred, columns=[f"pred_{c}" for c in CLASS_ORDER])
    for i, c in enumerate(CLASS_ORDER):
        out[f"true_{c}"] = Y_val[:, i]
    out.to_csv(VAL_PREDS_PATH, index=False)
    print(f"[ok] Saved GBM validation predictions → {VAL_PREDS_PATH}")

if __name__ == "__main__":
    main()

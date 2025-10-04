#!/usr/bin/env python3
import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from joblib import dump

from core.features import Profile, build_feature_vector
from core.model import SoftmaxShareModel, CLASS_ORDER

BASE = Path(__file__).resolve().parents[1]
DATA_SYN = BASE / "data" / "synthetic"
MODEL_PATH = DATA_SYN / "softmax_model.json"
TRAIN = DATA_SYN / "home_train.csv"
VAL   = DATA_SYN / "home_val.csv"
OUT   = DATA_SYN / "residual_corrector.joblib"
META  = DATA_SYN / "residual_meta.txt"

IMPROVE_THRESH = 0.05  # pp improvement required to save corrector

def row_to_x(row):
    p = Profile(
        bill_aed=float(row["bill_aed"]), tariff=float(row["tariff"]),
        home_type=str(row["home_type"]), size=str(row["size"]),
        occupants=int(row["occupants"]), setpoint=int(row["setpoint"]),
        led_pct=int(row["led_pct"]),
    )
    extra = {"month": int(row.get("month", 0)), "cdd_proxy": float(row.get("cdd_proxy", 0.0))}
    return build_feature_vector(p, extra=extra)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True); return p

def baseline_logits(model: SoftmaxShareModel, X):
    P = np.vstack([list(model.predict_shares(X[i]).values()) for i in range(len(X))])
    eps=1e-6; logp = np.log(P+eps)
    return logp - logp.mean(axis=1, keepdims=True)

def mae_pp(P, Y):
    return (np.abs(P - Y).mean(axis=0) * 100.0)

def main():
    # Load data and baseline
    base = SoftmaxShareModel(); base.load(str(MODEL_PATH))
    train = pd.read_csv(TRAIN); val = pd.read_csv(VAL)

    X_train = np.vstack([row_to_x(r) for _, r in train.iterrows()])
    Y_train = train[[f"{c}_share" for c in CLASS_ORDER]].values
    X_val   = np.vstack([row_to_x(r) for _, r in val.iterrows()])
    Y_val   = val[[f"{c}_share" for c in CLASS_ORDER]].values

    # Baseline preds/MAE
    Z_val_base = baseline_logits(base, X_val)
    P_val_base = softmax(Z_val_base)
    base_mae_vec = mae_pp(P_val_base, Y_val)
    base_mae_avg = float(base_mae_vec.mean())
    print("Baseline Val MAE (pp):", {c: round(base_mae_vec[i],2) for i,c in enumerate(CLASS_ORDER)}, "Avg:", round(base_mae_avg,2))

    # Prepare residual targets
    Z_train_base = baseline_logits(base, X_train)
    Z_val_base   = baseline_logits(base, X_val)

    eps=1e-6
    T_train = np.log(Y_train+eps) - np.log(Y_train+eps).mean(axis=1, keepdims=True)
    T_val   = np.log(Y_val+eps)   - np.log(Y_val+eps).mean(axis=1, keepdims=True)

    R_train = T_train - Z_train_base
    R_val   = T_val   - Z_val_base

    candidates = []

    # Candidate 1: Ridge
    for alpha in [1e-3, 1e-2, 1e-1, 1.0]:
        rg = Ridge(alpha=alpha, random_state=42)
        rg.fit(X_train, R_train)
        Z_val_corr = Z_val_base + rg.predict(X_val)
        P_val_corr = softmax(Z_val_corr)
        mae_vec = mae_pp(P_val_corr, Y_val)
        avg = float(mae_vec.mean())
        candidates.append(("ridge", {"alpha":alpha}, rg, avg, mae_vec))

    # Candidate 2: small MLPs
    mlp_configs = [
        (32,), (64,), (64,32),
    ]
    for h in mlp_configs:
        mlp = MLPRegressor(hidden_layer_sizes=h, activation="relu", solver="adam",
                           alpha=1e-4, learning_rate_init=1e-3, max_iter=500,
                           random_state=42)
        mlp.fit(X_train, R_train)
        Z_val_corr = Z_val_base + mlp.predict(X_val)
        P_val_corr = softmax(Z_val_corr)
        mae_vec = mae_pp(P_val_corr, Y_val)
        avg = float(mae_vec.mean())
        candidates.append(("mlp", {"hidden":h}, mlp, avg, mae_vec))

    # Pick best
    best = min(candidates, key=lambda x: x[3])  # lowest avg MAE
    kind, params, model, best_avg, best_vec = best
    print(f"Best residual model: {kind} {params} | Val Avg MAE: {best_avg:.2f} (baseline {base_mae_avg:.2f})")

    # Save only if it beats baseline by threshold
    if base_mae_avg - best_avg >= IMPROVE_THRESH:
        dump(model, OUT)
        META.write_text(
            f"best={kind} params={params} "
            f"avg_mae={best_avg:.4f} "
            f"vec={{{', '.join([f'{c}:{best_vec[i]:.2f}' for i,c in enumerate(CLASS_ORDER)])}}} "
            f"baseline_avg={base_mae_avg:.4f}\n"
        )
        print(f"[ok] Saved residual corrector → {OUT}")
    else:
        # If an old file exists and is worse, remove it
        if OUT.exists():
            OUT.unlink()
        if META.exists():
            META.unlink()
        print("[info] No meaningful improvement; not saving a residual corrector.")

if __name__ == "__main__":
    main()

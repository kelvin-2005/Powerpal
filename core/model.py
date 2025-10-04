# core/model.py
import json
import numpy as np
from typing import Dict
from sklearn.linear_model import Ridge

# MUST match your data columns & app categories:
# ac_share, lighting_share, appliances_share, other_share
CLASS_ORDER = ["ac", "lighting", "appliances", "other"]

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    return p

class SoftmaxShareModel:
    """
    Predict 4 composition shares via linear regression on log-shares, then softmax.
    - Train: compute target logits t_i = log(s_i + eps) - mean(log(s+eps)) per row,
      fit 4 separate Ridge regressors to predict each t_i from X.
    - Predict: get 4 predicted logits, apply softmax -> shares that sum to 1.
    This avoids class collapse and handles continuous targets.
    """
    def __init__(self, alpha: float = 1.0):
        self.regs = {c: Ridge(alpha=alpha) for c in CLASS_ORDER}
        self.feature_mean = None
        self.feature_std = None
        self.fitted = False

    def _fit_scaler(self, X: np.ndarray):
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    def train(self, X: np.ndarray, Y_shares: np.ndarray):
        """
        X: (n,d)
        Y_shares: (n,4) with rows summing to 1
        """
        assert Y_shares.shape[1] == 4
        eps = 1e-6
        self._fit_scaler(X)
        Xs = self._apply_scaler(X)

        # Target logits (centered)
        log_s = np.log(Y_shares + eps)
        log_s_centered = log_s - log_s.mean(axis=1, keepdims=True)

        # Fit one regressor per component
        for i, c in enumerate(CLASS_ORDER):
            self.regs[c].fit(Xs, log_s_centered[:, i])

        self.fitted = True

    def predict_shares(self, x: np.ndarray) -> Dict[str, float]:
        """
        x: (d,) feature vector
        returns dict of shares summing to ~1.
        """
        if not self.fitted and (self.feature_mean is None or self.feature_std is None):
            raise RuntimeError("Model not fitted/loaded.")

        xs = self._apply_scaler(x.reshape(1, -1))
        z = np.array([self.regs[c].predict(xs)[0] for c in CLASS_ORDER])
        p = _softmax(z)
        # Clip tiny values and renormalize for numeric safety
        p = np.clip(p, 1e-8, None)
        p = p / p.sum()
        return {c: float(p[i]) for i, c in enumerate(CLASS_ORDER)}

    def save(self, path: str):
        obj = {
            "coef": {c: self.regs[c].coef_.tolist() for c in CLASS_ORDER},
            "intercept": {c: float(self.regs[c].intercept_) for c in CLASS_ORDER},
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "class_order": CLASS_ORDER,  # helpful for future-proofing
        }
        with open(path, "w") as f:
            json.dump(obj, f)

    def load(self, path: str):
        with open(path, "r") as f:
            obj = json.load(f)
        # Optional: validate class order in saved model
        if "class_order" in obj:
            saved_order = obj["class_order"]
            if saved_order != CLASS_ORDER:
                raise ValueError(f"Saved model class_order {saved_order} != current {CLASS_ORDER}. "
                                 f"Update CLASS_ORDER or retrain.")
        self.feature_mean = np.array(obj["feature_mean"])
        self.feature_std = np.array(obj["feature_std"])
        # Rebuild regressors and set coefs
        self.regs = {c: Ridge(alpha=1.0) for c in CLASS_ORDER}
        for c in CLASS_ORDER:
            self.regs[c].coef_ = np.array(obj["coef"][c])
            self.regs[c].intercept_ = obj["intercept"][c]
        self.fitted = True

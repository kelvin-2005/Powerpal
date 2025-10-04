# core/model_gbm.py
import json
import numpy as np
import lightgbm as lgb
from typing import Dict, List

from core.model import CLASS_ORDER

class GBMShareModel:
    """
    LightGBM on centered logits -> softmax. Supports:
      - type: "lgbm_centered_logits" (single model per class)
      - type: "lgbm_centered_logits_ensemble" (list of models per class)
    """
    def __init__(self):
        self.mode = None  # "single" | "ensemble"
        self.models: Dict[str, List[lgb.Booster] | lgb.Booster] = {}
        self.best_iters = {}
        self.feature_mean = None
        self.feature_std = None
        self.loaded = False

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    def load(self, path: str):
        with open(path, "r") as f:
            obj = json.load(f)

        t = obj.get("type")
        if t == "lgbm_centered_logits":
            self.mode = "single"
            self.feature_mean = np.array(obj["feature_mean"], dtype=float)
            self.feature_std  = np.array(obj["feature_std"],  dtype=float) + 1e-12
            self.best_iters   = {k: int(v) for k, v in obj.get("best_iterations", {}).items()}
            self.models = {}
            for c in CLASS_ORDER:
                booster = lgb.Booster(model_str=obj["models"][c])
                self.models[c] = booster
                if c not in self.best_iters:
                    self.best_iters[c] = booster.best_iteration or booster.current_iteration()

        elif t == "lgbm_centered_logits_ensemble":
            self.mode = "ensemble"
            self.feature_mean = np.array(obj["feature_mean"], dtype=float)
            self.feature_std  = np.array(obj["feature_std"],  dtype=float) + 1e-12
            self.best_iters   = {c: [int(x) for x in obj["best_iterations"][c]] for c in CLASS_ORDER}
            self.models = {}
            for c in CLASS_ORDER:
                boosters = [lgb.Booster(model_str=s) for s in obj["models_ensemble"][c]]
                self.models[c] = boosters
        else:
            raise ValueError(f"Unsupported model type: {t}")

        self.loaded = True
        return self

    def predict_shares(self, x: np.ndarray) -> Dict[str, float]:
        if not self.loaded:
            raise RuntimeError("GBMShareModel not loaded.")
        xs = self._apply_scaler(x.reshape(1, -1))

        if self.mode == "single":
            z = np.array([self.models[c].predict(xs, num_iteration=self.best_iters.get(c, 0))[0] for c in CLASS_ORDER], dtype=float)
        else:
            zs = []
            for c in CLASS_ORDER:
                boosters: List[lgb.Booster] = self.models[c]  # type: ignore
                iters = self.best_iters[c]
                preds = [booster.predict(xs, num_iteration=iters[k])[0] for k, booster in enumerate(boosters)]
                zs.append(float(np.mean(preds)))
            z = np.array(zs, dtype=float)

        z = z - z.mean()
        p = np.exp(z)
        p = p / p.sum()
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()
        return {c: float(p[i]) for i, c in enumerate(CLASS_ORDER)}

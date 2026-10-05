"""Forecasting models used by the admission-control simulator."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import joblib

from .data import FEATURE_COLUMNS, RESOURCE_COLUMNS, MaternaProfileStore
from .config import SimulatorConfig


class BaseForecaster:
    def forecast(self, history_features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def batch_forecast(self, histories: list[np.ndarray]) -> np.ndarray:
        if not histories:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([self.forecast(h) for h in histories])


class PersistenceForecaster(BaseForecaster):
    """Predicts the next resource factors using the last observed resource factors."""
    def forecast(self, history_features: np.ndarray) -> np.ndarray:
        if history_features.size == 0:
            return np.zeros(3, dtype=float)
        return np.asarray(history_features[-1, :3], dtype=float)

    def batch_forecast(self, histories: list[np.ndarray]) -> np.ndarray:
        if not histories:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([self.forecast(h) for h in histories])


class OracleForecaster(BaseForecaster):
    """Placeholder oracle forecaster; actual future value is supplied by the simulator."""
    def forecast(self, history_features: np.ndarray) -> np.ndarray:
        return PersistenceForecaster().forecast(history_features)


@dataclass
class XGBoostForecaster(BaseForecaster):
    model: object
    window: int = 12
    clip: tuple[float, float] = (0.0, 1.0)

    def _make_row(self, history_features: np.ndarray) -> np.ndarray:
        if history_features is None or len(history_features) == 0:
            return np.zeros(self.window * len(FEATURE_COLUMNS), dtype=float)
        hist = np.asarray(history_features, dtype=float)
        if hist.shape[1] < len(FEATURE_COLUMNS):
            raise ValueError(f"Expected at least {len(FEATURE_COLUMNS)} features, got {hist.shape[1]}")
        if len(hist) < self.window:
            pad = np.repeat(hist[:1], self.window - len(hist), axis=0)
            hist = np.vstack([pad, hist])
        return hist[-self.window:, :len(FEATURE_COLUMNS)].reshape(-1)

    def forecast(self, history_features: np.ndarray) -> np.ndarray:
        x = self._make_row(history_features).reshape(1, -1)
        y = np.asarray(self.model.predict(x), dtype=float).reshape(-1)[:3]
        return np.clip(y, self.clip[0], self.clip[1])

    def batch_forecast(self, histories: list[np.ndarray]) -> np.ndarray:
        if not histories:
            return np.zeros((0, 3), dtype=float)
        X = np.vstack([self._make_row(h) for h in histories])
        y = np.asarray(self.model.predict(X), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 3)
        return np.clip(y[:, :3], self.clip[0], self.clip[1])

    @classmethod
    def load(cls, model_path: str | Path, window: int = 12) -> "XGBoostForecaster":
        return cls(model=joblib.load(model_path), window=window)

    def save(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)


def default_xgboost_params() -> dict:
    """Best validation parameters from the forecasting benchmark."""
    return {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 4,
    }


def train_xgboost_from_profiles(
    data_root: str | Path,
    model_path: str | Path,
    cfg: SimulatorConfig | None = None,
    max_train_windows: int = 250_000,
    max_profiles: int | None = None,
    n_estimators_override: int | None = None,
    seed: int = 42,
) -> XGBoostForecaster:
    """Train and save the XGBoost runtime forecaster.

    The model uses flattened windows of [CPU factor, memory factor, bandwidth
    factor, demand score] and predicts the next [CPU, memory, bandwidth] factors.
    """
    cfg = cfg or SimulatorConfig()
    rng = np.random.default_rng(seed)
    store = MaternaProfileStore(data_root, max_profiles=max_profiles)
    X, y = store.build_supervised_windows(
        window=cfg.lookback_window,
        horizon=cfg.horizon,
        max_windows=max_train_windows,
        split="split_80_20",
        split_value="train",
        rng=rng,
        max_profiles=max_profiles,
    )

    try:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except Exception as e:
        raise RuntimeError(
            "xgboost and scikit-learn are required. Install them using `pip install -r requirements.txt`."
        ) from e

    params = default_xgboost_params()
    params["random_state"] = seed
    if n_estimators_override is not None:
        params["n_estimators"] = int(n_estimators_override)
    base = XGBRegressor(**params)
    model = MultiOutputRegressor(base)
    model.fit(X, y)

    forecaster = XGBoostForecaster(model=model, window=cfg.lookback_window)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "model": "MultiOutputRegressor(XGBRegressor)",
        "features": FEATURE_COLUMNS,
        "targets": RESOURCE_COLUMNS,
        "window": cfg.lookback_window,
        "horizon": cfg.horizon,
        "max_train_windows": max_train_windows,
        "max_profiles": max_profiles,
        "params": params,
        "seed": seed,
        "n_samples": int(len(X)),
    }
    model_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    return forecaster


def load_or_train_xgboost(
    data_root: str | Path,
    model_path: str | Path,
    cfg: SimulatorConfig,
    max_train_windows: int = 250_000,
    max_profiles: int | None = None,
    n_estimators_override: int | None = None,
    seed: int = 42,
    force_train: bool = False,
    allow_smoke_model: bool = False,
) -> XGBoostForecaster:
    model_path = Path(model_path)
    if model_path.exists() and not force_train:
        meta_path = model_path.with_suffix(".metadata.json")
        if ("smoke" in model_path.name.lower()) and not allow_smoke_model:
            raise ValueError(
                f"Refusing to use a reduced-size check model for the main experiment: {model_path}. "
                "Pass allow_smoke_model=True only for a quick development check, or use --force-train-model."
            )
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                n_samples = int(meta.get("n_samples", 0))
                max_profiles_meta = meta.get("max_profiles")
                if n_samples and n_samples < 50000 and not allow_smoke_model:
                    raise ValueError(
                        f"Existing model appears too small for manuscript-scale evaluation "
                        f"(n_samples={n_samples}). Use --force-train-model or explicitly allow reduced-size check models."
                    )
            except ValueError:
                raise
            except Exception:
                pass
        return XGBoostForecaster.load(model_path, window=cfg.lookback_window)
    return train_xgboost_from_profiles(
        data_root=data_root,
        model_path=model_path,
        cfg=cfg,
        max_train_windows=max_train_windows,
        max_profiles=max_profiles,
        n_estimators_override=n_estimators_override,
        seed=seed,
    )

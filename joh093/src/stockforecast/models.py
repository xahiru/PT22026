"""Forecasting models.

Five forecasters share a minimal ``fit / predict`` interface:

* :class:`NaiveModel` - persistence baseline (last close).
* :class:`RidgeModel` - regularised linear regression on engineered features.
* :class:`RandomForestModel` - bagged decision-tree ensemble.
* :class:`GBModel` - gradient-boosted trees (HistGradientBoosting).
* :class:`LSTMModel` - two-layer LSTM network (PyTorch, optional dependency).

The four scikit-learn models always work. The LSTM is only constructed when
PyTorch is installed; otherwise it is reported as unavailable and the other
models still run end to end.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .errors import ModelNotAvailableError, ModelTrainingError

log = logging.getLogger(__name__)

# Optional PyTorch dependency -------------------------------------------------
try:  # pragma: no cover - exercised only on machines with torch installed
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Scikit-learn based forecasters
# --------------------------------------------------------------------------- #
class NaiveModel:
    """Persistence baseline: every prediction equals the last training close."""

    name = "naive"

    def __init__(self, **_: Any) -> None:
        self._last_close: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "NaiveModel":
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ModelTrainingError("Naive model received empty training target.")
        self._last_close = float(y_arr[-1])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._last_close is None:
            raise ModelTrainingError("Naive model must be fitted before prediction.")
        return np.full(len(X), self._last_close, dtype=float)


class _SklearnModel:
    """Common wrapper for scikit-learn pipelines."""

    name = "sklearn"
    _estimator: Any

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "_SklearnModel":
        try:
            self._estimator.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            raise ModelTrainingError(f"{self.name} failed to train: {exc}") from exc
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(X), dtype=float)


class RidgeModel(_SklearnModel):
    name = "ridge"

    def __init__(self, alpha: float = 1.0, **_: Any) -> None:
        self._estimator = Pipeline(
            steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )


class RandomForestModel(_SklearnModel):
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        self._estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class GBModel(_SklearnModel):
    name = "gradient_boosting"

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        l2_regularization: float = 1.0,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        self._estimator = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )


# --------------------------------------------------------------------------- #
# LSTM forecaster (optional)
# --------------------------------------------------------------------------- #
class LSTMModel(_SklearnModel):
    """Two-layer LSTM regressor over scaled price/feature sequences."""

    name = "lstm"

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ModelNotAvailableError(
                "PyTorch is not installed; the LSTM model is unavailable. "
                "Install requirements-optional.txt to enable it."
            )
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

        self._x_scaler: StandardScaler | None = None
        self._y_scaler: StandardScaler | None = None
        self._net: Any = None
        self._losses: list[float] = []
        self._feature_count: int | None = None

    # -- internal PyTorch net -------------------------------------------------
    def _build_net(self, input_size: int) -> Any:
        import torch.nn as nn

        hidden_size = self.hidden_size
        num_layers = self.num_layers
        dropout = self.dropout

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x: Any) -> Any:
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net()

    def _to_tensor(self, array: np.ndarray) -> Any:
        return torch.from_numpy(array.astype(np.float32))

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LSTMModel":
        if not TORCH_AVAILABLE:
            raise ModelNotAvailableError("PyTorch is not installed.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)

        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        X_scaled = self._x_scaler.fit_transform(X_arr)
        y_scaled = self._y_scaler.fit_transform(y_arr).ravel()

        self._feature_count = X_scaled.shape[1]
        self._net = self._build_net(self._feature_count)

        # Build sequences from the scaled training matrix.
        from .features import make_sequences

        X_seq, y_seq = make_sequences(X_scaled, y_scaled, self.seq_len)
        X_t = self._to_tensor(X_seq)
        y_t = self._to_tensor(y_seq)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = torch.nn.MSELoss()

        n = len(X_t)
        self._losses = []
        self._net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(idx)
            avg_loss = epoch_loss / n
            self._losses.append(avg_loss)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                log.info("LSTM epoch %d/%d - MSE %.5f", epoch + 1, self.epochs, avg_loss)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE or self._net is None:
            raise ModelTrainingError("LSTM model must be fitted before prediction.")

        X_arr = np.asarray(X, dtype=float)
        X_scaled = self._x_scaler.transform(X_arr)  # type: ignore[union-attr]

        from .features import make_sequences

        X_seq, _ = make_sequences(X_scaled, np.zeros(len(X_scaled)), self.seq_len)
        self._net.eval()
        with torch.no_grad():
            pred_scaled = self._net(self._to_tensor(X_seq)).cpu().numpy()
        pred = self._y_scaler.inverse_transform(  # type: ignore[union-attr]
            pred_scaled.reshape(-1, 1)
        ).ravel()

        # Recover the full-length alignment: first seq_len-1 rows have no
        # complete sequence and are reported as NaN.
        full = np.full(len(X_scaled), np.nan, dtype=float)
        full[self.seq_len - 1 :] = pred
        return full

    @property
    def losses(self) -> list[float]:
        return self._losses


def _as_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Normalise an input matrix to a DataFrame (tree/model path uses DataFrames)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


class ResidualModel:
    """Predict the residual over the persistence (lag-1) close price.

    The inner estimator models ``Close - Lag_1`` (the daily change), which is
    far more stationary than the absolute price. The final prediction is
    reconstructed as ``Lag_1 + predicted residual``, so the forecast carries
    the most recent price level instead of reverting to the training-range
    centroid. This avoids the tree-model extrapolation wall on trending series.
    """

    def __init__(self, inner: Any, lag_col: str = "Lag_1") -> None:
        self.inner = inner
        self.lag_col = lag_col
        self.name = getattr(inner, "name", "residual")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "ResidualModel":
        Xdf = _as_dataframe(X)
        residual = np.asarray(y, dtype=float) - Xdf[self.lag_col].to_numpy(dtype=float)
        self.inner.fit(Xdf, residual)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xdf = _as_dataframe(X)
        lag1 = Xdf[self.lag_col].to_numpy(dtype=float)
        pred_resid = np.asarray(self.inner.predict(Xdf), dtype=float)
        return lag1 + pred_resid

    @property
    def losses(self) -> list[float]:
        return getattr(self.inner, "losses", [])


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_MODEL_REGISTRY = {
    "naive": NaiveModel,
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GBModel,
    "lstm": LSTMModel,
}


def available_model_names() -> list[str]:
    """Return model names that can currently be constructed."""
    names = list(_MODEL_REGISTRY.keys())
    if not TORCH_AVAILABLE:
        names = [n for n in names if n != "lstm"]
    return names


def build_model(name: str, model_cfg: dict[str, Any], random_state: int = 42) -> Any:
    """Instantiate a forecaster by name.

    Raises :class:`ModelNotAvailableError` for the LSTM when PyTorch is missing,
    and :class:`ModelTrainingError` for an unknown model name.
    """
    if name not in _MODEL_REGISTRY:
        raise ModelTrainingError(f"Unknown model '{name}'.")
    params = {k: v for k, v in model_cfg.items() if k != "enabled" and k != "description"}
    params["random_state"] = random_state
    inner = _MODEL_REGISTRY[name](**params)

    # Tree ensembles and the LSTM model the stationary daily-change residual
    # over the lag-1 close; see :class:`ResidualModel`.
    if name in ("random_forest", "gradient_boosting", "lstm"):
        return ResidualModel(inner)
    return inner

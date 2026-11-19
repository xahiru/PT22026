"""Configuration loading and project path resolution.

The configuration is split across two YAML files: ``app.yaml`` (data, features,
forecast settings) and ``models.yaml`` (per-model hyper-parameters). Paths are
resolved relative to the project root so the package works regardless of the
current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"


@dataclass
class Config:
    """Typed view over the merged YAML configuration."""

    raw: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    @property
    def project(self) -> dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def features(self) -> dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def forecast(self) -> dict[str, Any]:
        return self.raw.get("forecast", {})

    def model(self, name: str) -> dict[str, Any]:
        """Return the hyper-parameter block for a model name."""
        return self.raw.get("models", {}).get(name, {})

    def enabled_models(self) -> list[str]:
        """Return model names whose block has ``enabled: true``."""
        return [
            name
            for name, block in self.raw.get("models", {}).items()
            if isinstance(block, dict) and block.get("enabled", True)
        ]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def load_config(config_dir: Path | str | None = None) -> Config:
    """Load and merge ``app.yaml`` and ``models.yaml`` into a :class:`Config`.

    Optional environment variables override the YAML defaults:
    ``STOCKFORECAST_DEFAULT_TICKER`` and ``STOCKFORECAST_SEED``.
    """
    base = Path(config_dir) if config_dir else CONFIG_DIR
    app_cfg = _read_yaml(base / "app.yaml")
    models_cfg = _read_yaml(base / "models.yaml")
    merged: dict[str, Any] = {**app_cfg, "models": models_cfg}

    env_ticker = os.environ.get("STOCKFORECAST_DEFAULT_TICKER")
    if env_ticker:
        merged.setdefault("data", {})["default_ticker"] = env_ticker
    env_seed = os.environ.get("STOCKFORECAST_SEED")
    if env_seed:
        try:
            merged.setdefault("data", {})["random_state"] = int(env_seed)
        except ValueError:
            pass

    return Config(raw=merged)


def ensure_directories() -> None:
    """Create runtime output directories if they do not exist."""
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

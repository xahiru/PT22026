"""Stock price forecasting package.

A small, modular time-series forecasting project for the Professional Training
mid-term assignment (YN3012170034).
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import Config, load_config
from .pipeline import PipelineResult, run_pipeline, train_and_evaluate

__all__ = ["Config", "PipelineResult", "load_config", "run_pipeline", "train_and_evaluate"]

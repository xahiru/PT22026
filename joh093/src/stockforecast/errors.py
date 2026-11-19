"""Custom exception hierarchy for the stock forecasting project.

Keeping project-specific errors in one place lets the UI and CLI translate
failures into readable messages instead of leaking stack traces to the user.
"""

from __future__ import annotations


class StockForecastError(Exception):
    """Base class for all project-specific errors."""


class DataSourceError(StockForecastError):
    """Raised when market data cannot be downloaded or retrieved."""


class InsufficientDataError(StockForecastError):
    """Raised when the available history is too short to train a model."""


class InvalidDataError(StockForecastError):
    """Raised when the supplied dataset fails validation."""


class ModelTrainingError(StockForecastError):
    """Raised when a model fails to fit on the prepared data."""


class ModelNotAvailableError(StockForecastError):
    """Raised when a requested model's optional dependency is missing."""

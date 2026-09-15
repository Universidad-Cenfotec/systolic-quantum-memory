"""Scalar zero-noise extrapolation utilities."""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, Sequence

import numpy as np


class ExtrapolationResult(NamedTuple):
    """Holds both the raw mathematical result and the [0,1]-bounded value."""

    raw: float
    bounded: float


class ZNEExtrapolator:
    """Extrapolate a scalar metric, such as final-register fidelity, to noise 0."""

    @staticmethod
    def extrapolate(
        noise_factors: Sequence[float],
        values: Sequence[float],
        method: str = "linear",
        **kwargs: Any,
    ) -> ExtrapolationResult:
        """Return the fitted value at noise factor zero.

        Only the scalar metric is extrapolated. Counts and probability
        distributions are intentionally never extrapolated directly.

        Returns an ``ExtrapolationResult`` with both the raw mathematical
        estimate and its ``[0, 1]``-bounded version.
        """
        factors = np.asarray(noise_factors, dtype=float)
        observations = np.asarray(values, dtype=float)

        if factors.ndim != 1 or observations.ndim != 1:
            raise ValueError("noise_factors and values must be one-dimensional")
        if len(factors) != len(observations):
            raise ValueError("noise_factors and values must have equal length")
        if len(factors) < 2:
            raise ValueError("at least two noise factors are required")
        if not np.all(np.isfinite(factors)) or not np.all(np.isfinite(observations)):
            raise ValueError("noise_factors and values must be finite")
        if len(np.unique(factors)) != len(factors):
            raise ValueError("noise_factors must be unique")
        if np.any(factors <= 0):
            raise ValueError("noise_factors must be positive")

        method = method.lower()
        if method == "linear":
            coefficients = np.polyfit(factors, observations, 1)
            estimate = np.polyval(coefficients, 0.0)
        elif method == "polynomial":
            if len(factors) < 3:
                raise ValueError("polynomial extrapolation requires at least three factors")
            coefficients = np.polyfit(factors, observations, len(factors) - 1)
            estimate = np.polyval(coefficients, 0.0)
        elif method == "exponential":
            asymptote = kwargs.get("asymptote", 0.5)
            adjusted_obs = np.asarray(observations, dtype=float) - asymptote
            
            # An exponential decay y = A * exp(-bx) requires all adjusted values to be > 0.
            # If the signal is within 3% of the completely mixed asymptote, it is dominated
            # by shot noise (for ~1024 shots, error is ~3%). Exponential log-fits on shot 
            # noise produce wild spikes (extrapolating to >1.0).
            if np.any(adjusted_obs <= 0.03):
                # Fall back to linear extrapolation for points deep in the noise floor
                coefficients = np.polyfit(factors, observations, 1)
                estimate = float(np.polyval(coefficients, 0.0))
            else:
                coefficients = np.polyfit(factors, np.log(adjusted_obs), 1)
                estimate = float(np.exp(np.polyval(coefficients, 0.0)) + asymptote)
        else:
            raise ValueError(f"unknown extrapolation method: {method}")

        raw_value = float(estimate)
        bounded_value = float(np.clip(raw_value, 0.0, 1.0))
        if raw_value != bounded_value:
            warnings.warn(
                f"ZNE extrapolation produced {raw_value:.6f}; "
                f"clipped value is {bounded_value:.6f}",
                RuntimeWarning,
                stacklevel=2,
            )
        return ExtrapolationResult(raw=raw_value, bounded=bounded_value)

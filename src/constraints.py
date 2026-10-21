"""Portfolio weight constraints.

A single value object describes every limit a portfolio must respect:

* ``min_weights`` -- a floor per asset. This is how a mandatory holding is
  expressed: a core position that must always be present in the portfolio.
* ``max_weight`` -- a ceiling per asset, which stops the optimizer from
  concentrating the whole portfolio in one name.
* ``max_volatility`` -- a cap on the portfolio's annualized volatility.

The constraints are validated against the asset universe before the solver
runs, so an infeasible request fails with a clear message instead of an
obscure solver error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.errors import ConstraintError


@dataclass(frozen=True)
class WeightConstraints:
    """Limits a portfolio must respect.

    Attributes:
        min_weights: Minimum weight per ticker. Tickers that are absent are
            unconstrained from below.
        max_weight: Maximum weight for any single ticker, or ``None`` for no
            ceiling.
        max_volatility: Maximum annualized portfolio volatility, or ``None``
            for no cap.
    """

    min_weights: dict[str, float] = field(default_factory=dict)
    max_weight: float | None = None
    max_volatility: float | None = None

    def is_active(self) -> bool:
        """Return whether any constraint is set."""
        return bool(self.min_weights) or self.max_weight is not None or self.max_volatility is not None

    def validate(self, tickers: list[str]) -> None:
        """Check that the constraints can be satisfied by ``tickers``.

        Args:
            tickers: The asset universe the portfolio is built from.

        Raises:
            ConstraintError: If any constraint is invalid or infeasible.
        """
        for ticker, weight in self.min_weights.items():
            if ticker not in tickers:
                raise ConstraintError(
                    f"minimum weight refers to {ticker!r}, which is not in the asset universe"
                )
            if not np.isfinite(weight):
                raise ConstraintError(f"minimum weight for {ticker!r} is not a finite number")
            if weight < 0.0:
                raise ConstraintError(
                    f"minimum weight for {ticker!r} must not be negative, got {weight}"
                )
            if weight > 1.0:
                raise ConstraintError(
                    f"minimum weight for {ticker!r} must not exceed 1, got {weight}"
                )

        total_minimum = float(sum(self.min_weights.values()))
        if total_minimum > 1.0 + 1e-12:
            raise ConstraintError(
                f"minimum weights sum to {total_minimum:.4f}, which cannot be invested"
            )

        if self.max_weight is not None:
            if not np.isfinite(self.max_weight) or self.max_weight <= 0.0:
                raise ConstraintError(
                    f"maximum weight must be a positive number, got {self.max_weight}"
                )
            if self.max_weight > 1.0 + 1e-12:
                raise ConstraintError(
                    f"maximum weight must not exceed 1, got {self.max_weight}"
                )
            for ticker, weight in self.min_weights.items():
                if weight > self.max_weight + 1e-12:
                    raise ConstraintError(
                        f"minimum weight for {ticker!r} ({weight}) exceeds the maximum "
                        f"weight ({self.max_weight})"
                    )
            if len(tickers) * self.max_weight < 1.0 - 1e-12:
                raise ConstraintError(
                    f"maximum weight {self.max_weight} leaves the portfolio unable to "
                    f"invest fully across {len(tickers)} assets"
                )

        if self.max_volatility is not None:
            if not np.isfinite(self.max_volatility) or self.max_volatility <= 0.0:
                raise ConstraintError(
                    f"maximum volatility must be a positive number, got {self.max_volatility}"
                )

    def min_weight_vector(self, tickers: list[str]) -> np.ndarray:
        """Return the minimum weight of each ticker as a vector.

        Args:
            tickers: The asset universe, in order.

        Returns:
            A vector of minimum weights aligned with ``tickers``.
        """
        return np.array([float(self.min_weights.get(ticker, 0.0)) for ticker in tickers])

    def upper_bounds(self, tickers: list[str]) -> np.ndarray:
        """Return the maximum weight of each ticker as a vector.

        Args:
            tickers: The asset universe, in order.

        Returns:
            A vector of upper bounds; ``inf`` where no ceiling applies.
        """
        if self.max_weight is None:
            return np.full(len(tickers), np.inf)
        return np.full(len(tickers), float(self.max_weight))

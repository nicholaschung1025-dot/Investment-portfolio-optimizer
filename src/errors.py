"""Shared exception hierarchy for the optimizer.

Every error raised by this package derives from :class:`PortfolioError`, so a
caller can either catch the whole family or narrow down to one specific
failure mode. The optimizer re-exports :class:`OptimizationError` so that
``from src.optimizer import OptimizationError`` keeps working.
"""

from __future__ import annotations


class PortfolioError(RuntimeError):
    """Base class for every error raised by this package."""


class OptimizationError(PortfolioError):
    """Raised when portfolio inputs are invalid or the solver fails to converge."""


class CovarianceError(PortfolioError):
    """Raised when a covariance matrix cannot be estimated or is not PSD."""


class ConstraintError(PortfolioError):
    """Raised when weight constraints are inconsistent or unsatisfiable."""


class AllocationError(PortfolioError):
    """Raised when a target return cannot be reached by the risky portfolio."""

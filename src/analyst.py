"""Analyst views for the Black-Litterman posterior.

A consensus price target is a statement about where the market expects a
price to be in a year, which makes it a view on the expected return. It is
also a weak signal: sell-side targets are systematically optimistic and they
follow momentum, so a target that is far from where the market prices the
asset is more likely to be wrong than informative.

Two things happen here. The implied return is computed from the target, the
price and the dividend, and the view is dropped when it is an outlier
against the prior, so a single strange target cannot drag the whole
portfolio with it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.fundamentals import Fundamentals

MINIMUM_VIEWS_FOR_OUTLIER_DETECTION = 3


def analyst_implied_return(
    target: float, price: float, dividend_yield: float | None = None
) -> float:
    """Return the total return implied by an analyst price target.

    A price target is a price, so the dividend is added on top of the price
    gap to get a total return.

    Args:
        target: The analyst price target.
        price: The current price.
        dividend_yield: Annual dividend yield as a fraction of price.

    Returns:
        The implied annual total return.

    Raises:
        ValueError: If the price is not positive.
    """
    if price <= 0:
        raise ValueError(f"the price must be positive, got {price}")
    implied = (target / price) - 1.0
    if dividend_yield:
        implied += dividend_yield
    return implied


def analyst_views(
    fundamentals: Mapping[str, Fundamentals],
    prices: Mapping[str, float],
    prior: pd.Series | None = None,
    outlier_sigma: float = 3.0,
) -> dict[str, float]:
    """Turn analyst price targets into Black-Litterman views.

    Args:
        fundamentals: Normalized fundamentals per ticker.
        prices: Current price per ticker.
        prior: The prior expected return per ticker. When given, views that
            sit far from it are dropped as outliers.
        outlier_sigma: How many standard deviations away from the prior a view
            may sit before it is dropped.

    Returns:
        The implied return per ticker, for the tickers that have a usable
        target and survive the outlier check.
    """
    implied: dict[str, float] = {}
    for ticker, entry in fundamentals.items():
        if not entry.available or entry.analyst_target is None:
            continue
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        implied[ticker] = analyst_implied_return(
            entry.analyst_target, price, entry.dividend_yield
        )

    if prior is None or len(implied) < MINIMUM_VIEWS_FOR_OUTLIER_DETECTION:
        return implied

    deviations = np.array(
        [implied[ticker] - float(prior.get(ticker, 0.0)) for ticker in implied]
    )
    # The scale has to be robust: a plain standard deviation is inflated by
    # the very outlier it is meant to find, which lets it hide.
    center = float(np.median(deviations))
    scale = 1.4826 * float(np.median(np.abs(deviations - center)))
    if scale <= 0.0:
        return implied

    kept: dict[str, float] = {}
    for ticker, value in implied.items():
        deviation = abs(value - float(prior.get(ticker, 0.0)) - center)
        if deviation <= outlier_sigma * scale:
            kept[ticker] = value
    return kept

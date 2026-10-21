"""Quality filters for the candidate universe.

Some companies are not investable at any price: too small to survive a bad
quarter, too thinly traded to get in or out of, or so volatile that the
position would dominate the portfolio's risk. These rules cut them out
before the optimizer sees them, and they say why.

Two decisions matter here. A missing field is not a failed field: a company
whose volume Yahoo Finance does not report is not the same as a company with
no volume, so an unknown value is kept unless the rules say otherwise. And
the expected return is checked against the *raw* estimate, never against a
clamped one, so the filter can actually fail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.fundamentals import Fundamentals
from src.market import MarketMetadata


@dataclass(frozen=True)
class QualityRules:
    """The limits a company has to meet to be investable.

    Attributes:
        min_analyst_count: Minimum number of analyst opinions.
        min_market_cap_usd: Minimum market capitalization for US listings.
        min_market_cap_hkd: Minimum market capitalization for Hong Kong
            listings, in Hong Kong dollars.
        min_price: Minimum share price, in the listing currency.
        max_annual_volatility: Maximum annualized volatility.
        min_avg_volume: Minimum average daily share volume.
        min_expected_return: Lowest acceptable expected return.
        max_expected_return: Highest acceptable expected return.
        unknown_policy: ``"keep"`` keeps a ticker whose fundamental data is
            missing, ``"exclude"`` drops it.
    """

    min_analyst_count: int = 10
    min_market_cap_usd: float = 10e9
    min_market_cap_hkd: float = 80e9
    min_price: float = 10.0
    max_annual_volatility: float = 0.50
    min_avg_volume: float = 5_000_000
    min_expected_return: float = -0.20
    max_expected_return: float = 0.50
    unknown_policy: str = "keep"

    def market_cap_floor(self, currency: str) -> float:
        """Return the market capitalization floor for a listing currency."""
        if currency.upper() == "HKD":
            return self.min_market_cap_hkd
        return self.min_market_cap_usd


@dataclass(frozen=True)
class FilterResult:
    """The outcome of applying the quality rules.

    Attributes:
        kept: Tickers that passed every rule.
        dropped: Tickers that did not, with the reasons.
    """

    kept: list[str]
    dropped: list[tuple[str, list[str]]] = field(default_factory=list)


def apply_quality_filters(
    universe: Sequence[str],
    fundamentals: Mapping[str, Fundamentals],
    expected_returns: pd.Series,
    volatilities: pd.Series,
    rules: QualityRules,
    metadata: MarketMetadata | None = None,
) -> FilterResult:
    """Apply the quality rules to a candidate universe.

    Args:
        universe: Tickers to consider.
        fundamentals: Normalized fundamentals per ticker.
        expected_returns: The raw expected return per ticker, exactly as it
            is fed to the optimizer.
        volatilities: Annualized volatility per ticker.
        rules: The limits to apply.
        metadata: Market metadata, used for the listing currency.

    Returns:
        The :class:`FilterResult` with the surviving tickers and the reasons
        the others did not.
    """
    listing = metadata if metadata is not None else MarketMetadata()
    exclude_unknown = rules.unknown_policy == "exclude"

    kept: list[str] = []
    dropped: list[tuple[str, list[str]]] = []

    for ticker in universe:
        entry = fundamentals.get(ticker)
        reasons: list[str] = []

        analyst_count = entry.analyst_count if entry is not None else None
        if analyst_count is None:
            if exclude_unknown:
                reasons.append("analyst coverage unknown")
        elif analyst_count < rules.min_analyst_count:
            reasons.append(
                f"analyst coverage {analyst_count} below {rules.min_analyst_count}"
            )

        market_cap = entry.market_cap if entry is not None else None
        if market_cap is None:
            if exclude_unknown:
                reasons.append("market capitalization unknown")
        else:
            currency = listing.info(ticker).currency
            floor = rules.market_cap_floor(currency)
            if market_cap < floor:
                reasons.append(
                    f"market cap {market_cap / 1e9:,.1f}bn {currency} below "
                    f"{floor / 1e9:,.1f}bn {currency}"
                )

        price = entry.price if entry is not None else None
        if price is None:
            if exclude_unknown:
                reasons.append("price unknown")
        elif price < rules.min_price:
            reasons.append(f"price {price:,.2f} below {rules.min_price:,.2f}")

        avg_volume = entry.avg_volume if entry is not None else None
        if avg_volume is None:
            if exclude_unknown:
                reasons.append("average volume unknown")
        elif avg_volume < rules.min_avg_volume:
            reasons.append(
                f"average volume {avg_volume:,.0f} below {rules.min_avg_volume:,.0f}"
            )

        if ticker in volatilities.index:
            volatility = float(volatilities[ticker])
            if volatility > rules.max_annual_volatility:
                reasons.append(
                    f"volatility {volatility:.0%} above {rules.max_annual_volatility:.0%}"
                )
        elif exclude_unknown:
            reasons.append("volatility unknown")

        if ticker in expected_returns.index:
            expected = float(expected_returns[ticker])
            if not rules.min_expected_return <= expected <= rules.max_expected_return:
                reasons.append(
                    f"expected return {expected:.0%} outside "
                    f"[{rules.min_expected_return:.0%}, {rules.max_expected_return:.0%}]"
                )
        elif exclude_unknown:
            reasons.append("expected return unknown")

        if reasons:
            dropped.append((ticker, reasons))
        else:
            kept.append(ticker)

    return FilterResult(kept=kept, dropped=dropped)

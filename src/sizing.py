"""Position sizing: from target weights to a tradable order list.

An optimizer produces weights; a broker wants whole shares, and a Hong Kong
exchange wants whole board lots. Rounding therefore leaves cash behind, and
leaving it idle quietly drags the portfolio away from the weights that were
asked for. This module rounds to something tradable and then spends the
leftover on the position that is furthest from its target, which is where it
does the most good.

Costs are reported rather than assumed away: the estimated entry cost is a
function of the fee, so a portfolio can be judged net of the frictions
instead of pretending they do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.market import FxRates, MarketMetadata


@dataclass(frozen=True)
class Position:
    """One tradable position.

    Attributes:
        ticker: Ticker of the position.
        target_weight: Weight the optimizer asked for.
        shares: Number of shares to buy.
        board_lot: Size of one board lot for this listing.
        price: Price per share in the listing currency.
        currency: Currency the asset is quoted in.
        value_usd: Value of the position in US dollars.
        realized_weight: Weight the position actually has.
        drift: Realized weight minus target weight.
    """

    ticker: str
    target_weight: float
    shares: int
    board_lot: int
    price: float
    currency: str
    value_usd: float
    realized_weight: float
    drift: float


@dataclass(frozen=True)
class PositionPlan:
    """The positions to trade and what is left over.

    Attributes:
        positions: The positions, one per tradable ticker.
        invested_value: Total value of the positions, in US dollars.
        cash_value: Cash left uninvested, in US dollars.
        total_value: Invested value plus cash.
        estimated_cost: Estimated cost of putting the plan into effect.
        skipped: Tickers that were dropped because no price was available.
        warnings: Human-readable notes about positions that could not be
            bought as planned.
    """

    positions: list[Position]
    invested_value: float
    cash_value: float
    total_value: float
    estimated_cost: float
    skipped: list[str]
    warnings: list[str] = field(default_factory=list)


def _lot_cost(price_usd: float, board_lot: int) -> float:
    """Return the US dollar cost of one board lot."""
    return price_usd * board_lot


def size_positions(
    weights: pd.Series | dict[str, float],
    prices: dict[str, float],
    budget: float,
    metadata: MarketMetadata | None = None,
    fx_rates: FxRates | None = None,
    fee_bps: float = 0.0,
) -> PositionPlan:
    """Turn target weights into a tradable position list.

    Args:
        weights: Target weight per ticker.
        prices: Latest price per ticker, in the listing currency.
        budget: Total amount to invest.
        metadata: Market metadata, used for the currency and board lot.
        fx_rates: Exchange rates used to value the positions in US dollars.
        fee_bps: One-way trading cost in basis points.

    Returns:
        The :class:`PositionPlan` to trade.
    """
    if budget <= 0:
        raise ValueError(f"the budget must be positive, got {budget}")

    listing = metadata if metadata is not None else MarketMetadata()
    rates = fx_rates if fx_rates is not None else FxRates.default()

    target_weights = (
        weights.to_dict() if isinstance(weights, pd.Series) else dict(weights)
    )

    skipped: list[str] = []
    entries: list[dict[str, object]] = []
    for ticker, weight in target_weights.items():
        if weight is None or weight <= 0.0:
            continue
        price = prices.get(ticker)
        if price is None or price <= 0.0:
            skipped.append(ticker)
            continue

        info = listing.info(ticker)
        price_usd = rates.to_usd(price, info.currency)
        target_value = float(weight) * budget
        raw_shares = int(target_value // price_usd)
        shares = (raw_shares // info.board_lot) * info.board_lot
        entries.append(
            {
                "ticker": ticker,
                "target_weight": float(weight),
                "target_value": target_value,
                "shares": shares,
                "board_lot": info.board_lot,
                "price": float(price),
                "currency": info.currency,
                "price_usd": price_usd,
                "value_usd": shares * price_usd,
            }
        )

    cash = budget - sum(float(entry["value_usd"]) for entry in entries)

    # Only the rounding residual may be recycled. Cash the caller never
    # allocated to a position is meant to stay in cash -- spending it would
    # quietly turn a half-invested portfolio into a full one.
    allocated = sum(float(entry["target_value"]) for entry in entries)
    unallocated = max(0.0, budget - allocated)
    recyclable = max(0.0, cash - unallocated)

    # Spend the residual on the position that is furthest from its target,
    # as long as buying one lot brings that position closer. That is where a
    # lot of cash does the most good, and it never leaves the portfolio
    # further from its targets than idle cash would.
    while recyclable > 1e-9 and entries:
        lot_costs = {
            id(entry): _lot_cost(float(entry["price_usd"]), int(entry["board_lot"]))
            for entry in entries
        }
        gaps = {
            id(entry): float(entry["target_value"]) - float(entry["value_usd"])
            for entry in entries
        }
        candidates = [
            entry
            for entry in entries
            if lot_costs[id(entry)] <= recyclable
            and gaps[id(entry)] > lot_costs[id(entry)] / 2.0
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda entry: (gaps[id(entry)], -lot_costs[id(entry)], str(entry["ticker"])),
        )
        lot_cost = lot_costs[id(best)]
        best["shares"] = int(best["shares"]) + int(best["board_lot"])
        best["value_usd"] = float(best["value_usd"]) + lot_cost
        cash -= lot_cost
        recyclable -= lot_cost

    positions: list[Position] = []
    for entry in entries:
        value = float(entry["value_usd"])
        realized = value / budget
        positions.append(
            Position(
                ticker=str(entry["ticker"]),
                target_weight=float(entry["target_weight"]),
                shares=int(entry["shares"]),
                board_lot=int(entry["board_lot"]),
                price=float(entry["price"]),
                currency=str(entry["currency"]),
                value_usd=value,
                realized_weight=realized,
                drift=realized - float(entry["target_weight"]),
            )
        )

    invested = sum(position.value_usd for position in positions)
    warnings: list[str] = []
    for entry in entries:
        if int(entry["shares"]) == 0:
            lot_cost = _lot_cost(float(entry["price_usd"]), int(entry["board_lot"]))
            warnings.append(
                f"{entry['ticker']}: one board lot costs ${lot_cost:,.2f}, more than the "
                f"${float(entry['target_value']):,.2f} allocated to it, so the position "
                f"could not be bought"
            )
    return PositionPlan(
        positions=positions,
        invested_value=invested,
        cash_value=cash,
        total_value=invested + cash,
        estimated_cost=invested * fee_bps / 10_000.0,
        skipped=skipped,
        warnings=warnings,
    )

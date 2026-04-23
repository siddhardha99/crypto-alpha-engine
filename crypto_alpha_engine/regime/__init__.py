"""Regime tagging and per-regime metric breakdown."""

from __future__ import annotations

import pandas as pd

from crypto_alpha_engine.regime.tagger import (
    tag_funding,
    tag_trend,
    tag_volatility,
)

__all__ = [
    "build_default_labels",
    "tag_funding",
    "tag_trend",
    "tag_volatility",
]


def build_default_labels(
    *,
    close_for_trend: pd.Series,
    close_for_vol: pd.Series | None = None,
    funding_rate: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Convenience: apply the three SPEC §9 taggers with default parameters.

    The engine does NOT call this internally — regime labels are
    caller-supplied so the engine stays blind to which column is
    "the BTC close" etc. This helper is a one-liner for callers who
    want the default configuration.

    Args:
        close_for_trend: Daily close used for trend classification.
            Typically BTC daily close per SPEC §9.
        close_for_vol: Close used for volatility classification.
            Defaults to ``close_for_trend`` if omitted.
        funding_rate: Optional perp funding-rate series. If supplied,
            the returned dict includes a ``"funding"`` key.

    Returns:
        ``dict[str, pd.Series]`` keyed by dimension — ``"trend"`` and
        ``"vol"`` always present, ``"funding"`` only when funding_rate
        was passed.

    Example:
        >>> labels = build_default_labels(close_for_trend=btc_daily)
        >>> # Pass as regime_labels= to run_backtest:
        >>> # result = run_backtest(regime_labels=labels, ...)
    """
    if close_for_vol is None:
        close_for_vol = close_for_trend
    labels: dict[str, pd.Series] = {
        "trend": tag_trend(close_for_trend),
        "vol": tag_volatility(close_for_vol),
    }
    if funding_rate is not None:
        labels["funding"] = tag_funding(funding_rate)
    return labels

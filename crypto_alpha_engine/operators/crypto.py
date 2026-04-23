"""Crypto-specific operators — domain-vocabulary kernels.

These exist to give factor authors AST-level names that match their
domain (``funding_z`` reads naturally; ``ts_zscore`` of a funding
Series is the same math but less ergonomic in a factor spec). Every
kernel is a pure function of Series + window, like the generic
time-series operators; the Phase 4 compiler is the layer that resolves
AST symbol arguments to the right Series.

9 operators land here, matching SPEC §6.

Design note
-----------

Most crypto operators delegate to a generic timeseries kernel
(``ts_zscore``, ``ts_pct_change``, ``ts_corr``). The wrapper is NOT
redundant — it's the AST-level vocabulary that makes factors readable.
When a factor author writes ``FactorNode("funding_z", args=...)``,
they're expressing intent; ``ts_zscore`` applied to the funding series
would encode the same computation but bury the meaning.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_engine.operators.registry import register_operator
from crypto_alpha_engine.operators.timeseries import (
    ts_corr,
    ts_mean,
    ts_pct_change,
    ts_zscore,
)

# ---------------------------------------------------------------------------
# Funding / OI
# ---------------------------------------------------------------------------


@register_operator("funding_z", arg_types=("series", "int"))
def funding_z(funding: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of the funding-rate series.

    High positive values → the market is paying hard to be long
    (euphoric / crowded). High negative values → paying to be short
    (capitulation / crowded short). The z-score normalises across
    regimes where the absolute funding level drifts.
    """
    result: pd.Series = ts_zscore(funding, window)
    return result


@register_operator("oi_change", arg_types=("series", "int"))
def oi_change(oi: pd.Series, window: int) -> pd.Series:
    """Percent change in open interest over ``window`` bars.

    Combined with returns, captures whether a move is driven by new
    positioning (``oi_change > 0``) or position unwind (``< 0``).
    """
    result: pd.Series = ts_pct_change(oi, window)
    return result


# ---------------------------------------------------------------------------
# Fear & Greed (sentiment)
# ---------------------------------------------------------------------------


@register_operator("fear_greed", arg_types=("series", "int"))
def fear_greed(fg: pd.Series, window: int) -> pd.Series:
    """Rolling mean of the Fear & Greed index — a smoothed sentiment signal.

    The raw F&G is noisy day-to-day; this wrapper exposes the
    rolling-mean form that factor authors typically want.
    """
    result: pd.Series = ts_mean(fg, window)
    return result


# ---------------------------------------------------------------------------
# Macro-change family — all shape-identical, aliased by data source
# ---------------------------------------------------------------------------


@register_operator("btc_dominance_change", arg_types=("series", "int"))
def btc_dominance_change(dominance: pd.Series, window: int) -> pd.Series:
    """Percent change in BTC dominance over ``window`` bars.

    Rising dominance often coincides with BTC outperforming altcoins;
    falling dominance with alt-season regimes.
    """
    result: pd.Series = ts_pct_change(dominance, window)
    return result


@register_operator("stablecoin_mcap_change", arg_types=("series", "int"))
def stablecoin_mcap_change(mcap: pd.Series, window: int) -> pd.Series:
    """Percent change in total stablecoin market cap — a crude dry-powder proxy."""
    result: pd.Series = ts_pct_change(mcap, window)
    return result


@register_operator("active_addresses_change", arg_types=("series", "int"))
def active_addresses_change(addrs: pd.Series, window: int) -> pd.Series:
    """Percent change in unique active addresses — on-chain activity signal."""
    result: pd.Series = ts_pct_change(addrs, window)
    return result


@register_operator("hashrate_change", arg_types=("series", "int"))
def hashrate_change(rate: pd.Series, window: int) -> pd.Series:
    """Percent change in BTC network hashrate over ``window`` bars.

    Miner-capitulation moves show up as sharp negative spikes; secular
    bull phases as positive trends.
    """
    result: pd.Series = ts_pct_change(rate, window)
    return result


@register_operator("dxy_change", arg_types=("series", "int"))
def dxy_change(dxy: pd.Series, window: int) -> pd.Series:
    """Percent change in the US Dollar Index — a macro risk-off / risk-on signal."""
    result: pd.Series = ts_pct_change(dxy, window)
    return result


# ---------------------------------------------------------------------------
# Cross-series: correlation with SPY
# ---------------------------------------------------------------------------


@register_operator("spy_correlation", arg_types=("series", "series", "int"))
def spy_correlation(asset: pd.Series, spy: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between ``asset`` and SPY over ``window``.

    Captures the risk-asset regime: near 1.0 → crypto moves with
    equities; near 0 or negative → decoupling.
    """
    result: pd.Series = ts_corr(asset, spy, window)
    return result

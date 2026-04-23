"""Research workflow: iterating on a momentum factor — an honest failure.

This example walks through a real research loop against 2017-2026 BTC
daily bars from Coinbase:

    v1  — baseline momentum z-score
    v2  — EMA-smoothed version (hypothesis: "smoothing reduces whipsaw")
    compare via the ledger; interpret the result

Spoiler: v2's net Sharpe is LOWER than v1's despite halving the fees.
The point of the example is to show what you actually do when an
iteration fails — inspect the ledger, read the delta, update the
mental model. A research engine that only shows happy paths is
teaching you to lie to yourself.

Prerequisite
------------

Real data must be downloaded first::

    uv run python scripts/download_all_data.py

The script precondition-checks the parquet files and exits with an
actionable error if they're missing.

Usage
-----

Runs as a plain Python script::

    uv run python examples/research_workflow.py

The ``# %%`` markers make each logical step a cell in VS Code,
PyCharm, or any Jupytext-compatible editor so you can step through
them. Cells are roughly one idea per cell.
"""

# %% Imports + precondition check
from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_alpha_engine.backtest.engine import run_backtest
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.ledger.ledger import Ledger
from crypto_alpha_engine.regime import build_default_labels
from crypto_alpha_engine.types import (
    CostModel,
    Factor,
    FactorNode,
    WalkForwardConfig,
)

REPO = Path(__file__).resolve().parents[1]
BTC_DAILY = REPO / "data/ohlcv/coinbase_spot/BTC_USD_1d.parquet"
FUNDING_8H = REPO / "data/funding/bitmex_perp/BTC_USD_BTC_8h.parquet"

if not BTC_DAILY.exists() or not FUNDING_8H.exists():
    raise SystemExit(
        "This example needs real data. Run `uv run python "
        "scripts/download_all_data.py` first — ~45s, one-time."
    )

# %% Load real data (2017-2026 daily BTC close + BitMEX 8h funding)
close = pd.read_parquet(BTC_DAILY).set_index("timestamp").sort_index()["close"].astype(float)
funding_8h = pd.read_parquet(FUNDING_8H).set_index("timestamp")["funding_rate"].sort_index()
funding = funding_8h.resample("1D").sum().reindex(close.index).fillna(0.0)

# %% SPEC §9 regime labels — trend, vol, funding
regime_labels = build_default_labels(
    close_for_trend=close,
    funding_rate=funding,
)

# %% Ledger — fresh for this example run
ledger_path = REPO / "examples_ledger.jsonl"
ledger_path.unlink(missing_ok=True)
ledger = Ledger(ledger_path)

# %% Common run_backtest config — defined once so v1/v2 are comparable
COMMON = {
    "features": {"BTC/USD|close": close},
    "prices": close,
    "regime_labels": regime_labels,
    "funding_rate": funding,
    "feature_source_names": {"BTC/USD|close": "coinbase_spot"},
    "splits": DataSplits(train_end=close.index[-100], validation_end=close.index[-50]),
    "cost_model": CostModel(),
    "walk_forward_config": WalkForwardConfig(),
    "ledger": ledger,
    "freq": "1D",
    "min_test_bars": 60,
}

# %% Hypothesis v1 — "momentum-z predicts near-term returns"
# Factor: 72d z-score of 24d pct-change. The README's Tier 2 factor.
v1 = Factor(
    name="v1_zscore_momentum",
    description="72d z-score of 24d pct-change",
    hypothesis="Recent returns predict near-term returns",
    root=FactorNode(
        operator="ts_zscore",
        args=(
            FactorNode(operator="ts_pct_change", args=("BTC/USD|close", 24)),
            72,
        ),
    ),
)

# %% Backtest v1
r1 = run_backtest(factor=v1, **COMMON)
print("--- v1 baseline ---")
print(f"  Sharpe (net):            {r1.sharpe:.2f}")
print(f"  Max drawdown:            {r1.max_drawdown:.2%}")
print(f"  Trades:                  {r1.n_trades}")
print(f"  Fees:                    ${r1.total_fees_paid:,.0f}")
print(f"  Bull vs bear Sharpe:     {r1.bull_sharpe:.2f} / {r1.bear_sharpe:.2f}")

# %% Observation — strong regime asymmetry, many small trades
# Bull Sharpe ≫ bear Sharpe AND the gross-vs-net gap flags cost drag.
# Two plausible v2 directions: (a) filter out bear regime, (b) reduce
# turnover by smoothing the signal. We try (b) — it's the cleaner
# hypothesis because it doesn't require runtime regime labels in the
# signal path.

# %% Hypothesis v2 — "EMA-smoothing reduces whipsaw → higher net Sharpe"
# Factor: 14d EMA of v1's z-score output. Same base signal, filtered.
v2 = Factor(
    name="v2_ema_smoothed",
    description="14d EMA of the v1 z-score signal",
    hypothesis="Smoothing reduces whipsaw and improves net Sharpe",
    root=FactorNode(
        operator="ts_ema",
        args=(
            FactorNode(
                operator="ts_zscore",
                args=(
                    FactorNode(
                        operator="ts_pct_change",
                        args=("BTC/USD|close", 24),
                    ),
                    72,
                ),
            ),
            14,
        ),
    ),
)

# %% Backtest v2 — same ledger, same COMMON config
r2 = run_backtest(factor=v2, **COMMON)
print("\n--- v2 EMA-smoothed ---")
print(f"  Sharpe (net):            {r2.sharpe:.2f}")
print(f"  Max drawdown:            {r2.max_drawdown:.2%}")
print(f"  Trades:                  {r2.n_trades}")
print(f"  Fees:                    ${r2.total_fees_paid:,.0f}")
print(f"  Bull vs bear Sharpe:     {r2.bull_sharpe:.2f} / {r2.bear_sharpe:.2f}")

# %% Compare v1 vs v2 via the ledger
# pd.DataFrame constructor on ledger entries gives us a sortable table.
rows = []
for entry in ledger.read_all():
    rows.append(
        {
            "factor": entry.factor.name,
            "sharpe": entry.result.sharpe,
            "max_dd": entry.result.max_drawdown,
            "trades": entry.result.n_trades,
            "fees": entry.result.total_fees_paid,
            "bull": entry.result.bull_sharpe,
            "bear": entry.result.bear_sharpe,
        }
    )
comparison = pd.DataFrame(rows).set_index("factor")
print("\n--- ledger comparison ---")
print(comparison.round(2).to_string())

# %% Interpret: the iteration failed, and here's what to take away
# v2 cut fees roughly in half but net Sharpe dropped too. The EMA
# attenuated the signal along with the noise — sharp z-score
# transitions were doing real work, and smoothing them muted the
# edge without a proportional cost benefit. Next iteration should
# test: (a) a different denoising approach (e.g., rank instead of
# z-score), (b) a regime-conditional signal rule that trades only
# bull/crab regimes via signal_rule= kwarg, or (c) leave momentum
# alone and test a complementary signal (funding z-score).
#
# What the ledger gives us
#   - Both experiments are logged. When a third factor is tested,
#     its DSR multiple-testing correction will account for both
#     (sharpe_variance_across_trials flows from ledger automatically).
#   - Running v2 on this ledger didn't trip the duplicate check
#     because v2's AST (ts_ema wrapping v1) is structurally ~0.67
#     similar to v1 — below the 0.7 threshold. A near-duplicate
#     (e.g., same factor with window=73 instead of 72) would be
#     rejected by run_backtest.
#   - The gross/net Sharpe split, regime breakdown, and DSR are
#     scalars on BacktestResult. No way to access the underlying
#     equity curve — that's Principle 1 (sealed engine) by
#     construction, not by convention.

# %% Cleanup (optional) — remove the example ledger
# Uncomment to delete after this run:
# ledger_path.unlink(missing_ok=True)

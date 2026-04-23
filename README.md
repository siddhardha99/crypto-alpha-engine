# crypto-alpha-engine

A **sealed, cheat-proof backtesting engine** for crypto quantitative
research.

> **Status — v1.0.** Public release gate passed. See
> [`CHANGELOG.md`](CHANGELOG.md) for what ships, what's deliberately
> excluded, and known limitations.

## What it is

`crypto-alpha-engine` is a research tool. You submit a factor, you
get back a `BacktestResult`. The engine enforces:

1. **No lookahead bias** — two independent causality checks (AST
   whitelist + runtime perturbation) through the public API.
2. **Mandatory costs** — every backtest pays fees + slippage; there
   is no code path that silently skips cost application.
3. **Sealed output** — `BacktestResult` carries aggregate scalars
   only. No equity curves, no trade timestamps, no per-bar returns
   ever leak to user code.
4. **Walk-forward only** — no single-period, no in-sample-only
   evaluations.
5. **Experiment ledger** — every run is logged; Deflated Sharpe
   is computed against the ledger's real trial count and variance,
   not against a trust-me number.
6. **Regime breakdown by default** — every result includes trend /
   volatility / funding per-regime Sharpes. You can't opt out.

The project targets quant researchers who want to prototype
crypto-market strategies without accidentally fitting the future.

## Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
  (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git

## Install

```bash
git clone https://github.com/siddhardha99/crypto-alpha-engine.git
cd crypto-alpha-engine
uv sync --extra dev        # create .venv, install runtime + dev deps
uv run pytest              # sanity check: ~850 tests must pass
```

---

## Tier 1: 30-second quickstart (committed real data)

Runs a single-fold simulation against one month of real 2022 BTC
hourly data from the committed `tests/fixtures/` slice. No download,
no walk-forward — just a factor compiled on actual market bytes.

Copy into `quickstart_tier1.py`:

```python
from pathlib import Path

import pandas as pd

from crypto_alpha_engine.backtest.simulation import simulate_fold
from crypto_alpha_engine.factor.compiler import compile_factor
from crypto_alpha_engine.types import CostModel, Factor, FactorNode

# One month of real 1h BTC close (Coinbase spot, Jan 2022).
df = pd.read_parquet(
    "tests/fixtures/btc_usd_1h_coinbase_spot.parquet"
).set_index("timestamp")
close = df["close"].astype(float)

# Factor: 24h z-score of 4h pct-change (short-term momentum).
factor = Factor(
    name="momentum_z",
    description="24h z-score of 4h pct-change",
    hypothesis="Short-term momentum predicts near-term direction",
    root=FactorNode(
        operator="ts_zscore",
        args=(
            FactorNode(operator="ts_pct_change",
                       args=("BTC/USD|close", 4)),
            24,
        ),
    ),
)

# Compile + evaluate against the real close series.
compiled = compile_factor(factor)
factor_values = compiled({"BTC/USD|close": close})

# Generate entry/exit signals. The +1-bar shift is the causality
# gate — a signal observed at bar t executes at bar t+1.
entries = (factor_values > 0).fillna(False).shift(1).fillna(False).astype(bool)
exits = (factor_values < 0).fillna(False).shift(1).fillna(False).astype(bool)
entries.attrs["shifted"] = True   # engine rejects signals without this
exits.attrs["shifted"] = True

result = simulate_fold(
    entries=entries, exits=exits, close=close,
    cost_model=CostModel(), initial_cash=10_000.0, freq="1h",
)

print(f"Trades:          {result.n_trades}")
print(f"Fees paid:       ${result.fees_paid:.2f}")
print(f"Slippage paid:   ${result.slippage_paid:.2f}")
print(f"Gross return:    {(1 + result.gross_returns).prod() - 1:.2%}")
print(f"Net return:      {(1 + result.net_returns).prod() - 1:.2%}")
```

Run:

```bash
uv run python quickstart_tier1.py
```

Expected output (exact numbers depend on the committed fixture;
a representative run):

```
Trades:          87
Fees paid:       $1420.29
Slippage paid:   $710.15
Gross return:    -5.19%
Net return:      -26.97%
```

The gap between gross and net return is not a bug. It's the
engine refusing to flatter a naive factor.

---

## Tier 2: full quickstart (multi-year real data + walk-forward)

Runs the complete `run_backtest` pipeline — walk-forward across years
of real data, per-regime Sharpe breakdown, Deflated Sharpe against
the experiment ledger.

**Step 1. Download real market data** (~45s one-time; writes to
`data/`, which is `.gitignore`d):

```bash
uv run python scripts/download_all_data.py
```

**Step 2.** Copy into `quickstart_tier2.py`:

```python
from pathlib import Path

import pandas as pd

from crypto_alpha_engine.backtest.engine import run_backtest
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.ledger.ledger import Ledger
from crypto_alpha_engine.regime import build_default_labels
from crypto_alpha_engine.types import (
    CostModel, Factor, FactorNode, WalkForwardConfig,
)

# Multi-year daily BTC close + BitMEX 8h funding rate.
df = pd.read_parquet("data/ohlcv/coinbase_spot/BTC_USD_1d.parquet")
close = df.set_index("timestamp").sort_index()["close"].astype(float)

fdf = pd.read_parquet("data/funding/bitmex_perp/BTC_USD_BTC_8h.parquet")
funding_8h = fdf.set_index("timestamp")["funding_rate"].sort_index()
funding = funding_8h.resample("1D").sum().reindex(close.index).fillna(0.0)

# SPEC §9 regime labels, all three dimensions.
regime_labels = build_default_labels(
    close_for_trend=close, funding_rate=funding,
)

factor = Factor(
    name="momentum_z",
    description="72d z-score of 24d pct-change",
    hypothesis="Recent returns predict near-term returns",
    root=FactorNode(
        operator="ts_zscore",
        args=(
            FactorNode(operator="ts_pct_change",
                       args=("BTC/USD|close", 24)),
            72,
        ),
    ),
)

# Ledger — duplicate detection + DSR multi-testing correction.
# Delete to start fresh between runs.
ledger = Ledger(Path("my_ledger.jsonl"))

result = run_backtest(
    factor=factor,
    features={"BTC/USD|close": close},
    prices=close,
    regime_labels=regime_labels,
    funding_rate=funding,
    feature_source_names={"BTC/USD|close": "coinbase_spot"},
    splits=DataSplits(
        train_end=close.index[-100],
        validation_end=close.index[-50],
    ),
    cost_model=CostModel(),                     # SPEC §8 defaults
    walk_forward_config=WalkForwardConfig(),    # 24m train / 3m test / 1m step
    ledger=ledger,
    freq="1D",
    min_test_bars=60,
)

print(f"Sharpe (net, OOS):       {result.sharpe:.2f}")
print(f"Max drawdown:            {result.max_drawdown:.2%}")
print(f"Trades:                  {result.n_trades}")
print(f"Gross vs net Sharpe:     {result.gross_sharpe:.2f} → {result.net_sharpe:.2f}")
print(f"Bull regime Sharpe:      {result.bull_sharpe:.2f}")
print(f"Bear regime Sharpe:      {result.bear_sharpe:.2f}")
print(f"Deflated Sharpe:         {result.deflated_sharpe_ratio:.3f}")
print(f"Ledger entries:          {ledger.count_experiments()}")
```

Run:

```bash
uv run python quickstart_tier2.py
```

Representative output on 2017-2026 real BTC data:

```
Sharpe (net, OOS):       0.87
Max drawdown:            -82.69%
Trades:                  324
Gross vs net Sharpe:     0.96 → 0.87
Bull regime Sharpe:      6.12
Bear regime Sharpe:      0.09
Deflated Sharpe:         nan
Ledger entries:          1
```

Every field on `BacktestResult` is a scalar. There is no API that
returns the equity curve, individual trade timestamps, or per-bar
returns. This is Principle 1 (sealed engine) by construction.

`Deflated Sharpe: nan` is correct for a single-trial ledger — DSR's
multiple-testing correction needs prior trials to compute. Run
more experiments and the field populates.

---

## Where to go next

- [`SPEC.md`](SPEC.md) — the full build specification (what every
  module does and why)
- [`docs/methodology.md`](docs/methodology.md) — non-obvious design
  decisions with rationale
- [`docs/factor_design.md`](docs/factor_design.md) — the factor-AST
  contract
- [`examples/research_workflow.py`](examples/research_workflow.py) —
  a longer annotated workflow showing iteration and comparison
  against the ledger *(ships with v1.0 Phase 8 commit 2)*
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to add operators, data
  sources, and PRs

## Development

```bash
uv run pytest                    # full test suite
uv run ruff check .              # lint
uv run black --check .           # formatter check
uv run mypy crypto_alpha_engine tests scripts   # strict type check
```

Every PR must pass all four. Every new operator must ship with a
causality test. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).

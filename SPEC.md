# Crypto Alpha Engine — Build Specification

**Target:** A production-ready, open-source crypto quantitative research engine that rigorously backtests AI-generated trading strategies without allowing data leakage or common quant mistakes.

**For the AI coding agent:** This document is your complete specification. Read it fully before writing any code. Do not deviate from the architectural principles. All tests listed must pass before a module is considered complete.

---

## 1. Project Identity

**Name:** `crypto-alpha-engine` (the open-source public repo)
**Companion repo (private, not in this spec):** `crypto-alpha-lab` (the AI research agents)
**Language:** Python 3.11+
**License:** MIT
**Domain:** tradingwithcrypto.com (public site is separate)

**Mission:** Provide a sealed, cheat-proof backtesting environment for crypto strategies that eliminates the 12 classic quant mistakes by design, not by discipline.

---

## 2. Non-Negotiable Architectural Principles

These are the DNA of the project. If any code violates these, the code is wrong.

### Principle 1: The engine is sealed
The user (or AI researcher) never touches raw data directly. They submit a factor specification; the engine runs it and returns ONLY aggregate metrics. No trade-level data, no equity curves, no timestamps are ever returned to the caller.

### Principle 2: All operators are causal
Every time-series operator in the library uses only data from `[t-window, t]`. There is no operator that can reference `t+1` or later. Any operator that violates this must throw at build time.

### Principle 3: Data splits are sacred
Train/Validation/Test splits are enforced at the data-loading level. The test set is a physically separate file that the engine will not load during research mode. Test set access requires an explicit `reveal_test_set=True` flag that is logged.

### Principle 4: Every experiment is logged
No experiment is silently discarded. The experiment ledger is append-only, persists across runs, and is the denominator for Deflated Sharpe Ratio calculations.

### Principle 5: Costs are mandatory
Every backtest applies realistic fees, slippage, and funding costs. These are non-optional parameters with reasonable defaults; they cannot be set to zero.

### Principle 6: Walk-forward only
There is no single train/test backtest option. Every backtest is walk-forward by construction.

### Principle 7: Regime breakdown is automatic
Every BacktestResult includes per-regime Sharpe values. The user does not opt in.

### Principle 8: Idempotent data layer
Downloading the same data twice produces identical files. Data updates are append-only; historical bars are never overwritten.

---

## 3. Design Patterns to Follow

### Pattern: Immutable data contracts (frozen dataclasses)
All result types are `@dataclass(frozen=True)`. Once created, they cannot be modified.

### Pattern: Strategy as pure function
A factor is a pure function `(features_dict) -> pandas.Series`. No side effects, no I/O, no state.

### Pattern: Engine as orchestrator, not computer
The engine doesn't compute factors; it validates them, runs them through data, and summarizes results. Factor logic is entirely in the operator library.

### Pattern: Fail-loud validation
Every public function validates its inputs at entry. Invalid inputs raise specific exception types (not generic `Exception` or `ValueError` without a message).

### Pattern: Explicit over implicit
No hidden globals. No implicit timezone handling (always UTC). No "automatic" feature engineering. Every transformation is visible in the code path.

### Pattern: Protocol-based typing (Python's `Protocol`)
Instead of abstract base classes, use `typing.Protocol` for interfaces. Easier to test, easier to extend.

### Pattern: Registry for operators
The operator library is a dict `{"operator_name": function}` registered at import time. This allows the AI to compose factors using string names without executing arbitrary Python.

---

## 4. Complete Repository Structure

```
crypto-alpha-engine/
├── README.md
├── LICENSE                          # MIT
├── CONTRIBUTING.md
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── test.yml                 # CI: runs all tests on push
│
├── crypto_alpha_engine/             # The package
│   ├── __init__.py
│   ├── types.py                     # All frozen dataclasses
│   ├── exceptions.py                # All custom exceptions
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py               # Pandera schemas for data validation
│   │   ├── downloader.py            # ccxt wrappers, save to parquet
│   │   ├── loader.py                # Load splits, enforce immutability
│   │   ├── splits.py                # Train/val/test split logic
│   │   └── free_sources.py          # DefiLlama, Blockchain.com, etc.
│   │
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── registry.py              # The operator registry
│   │   ├── timeseries.py            # ts_mean, ts_std, etc.
│   │   ├── math.py                  # add, mul, log, etc.
│   │   ├── conditional.py           # if_else, greater_than, etc.
│   │   ├── crypto.py                # funding_z, oi_change, etc.
│   │   └── causality_check.py       # Verify no look-ahead
│   │
│   ├── factor/
│   │   ├── __init__.py
│   │   ├── ast.py                   # Factor AST representation
│   │   ├── parser.py                # Parse factor spec -> AST
│   │   ├── compiler.py              # AST -> executable function
│   │   ├── similarity.py            # AST similarity (for originality)
│   │   └── complexity.py            # Complexity metrics
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                # BacktestEngine main class
│   │   ├── walk_forward.py          # Walk-forward runner
│   │   ├── position_sizing.py       # Turn factor values into positions
│   │   ├── costs.py                 # Fees, slippage, funding
│   │   └── metrics.py               # Sharpe, Sortino, Calmar, IC, etc.
│   │
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── tagger.py                # Classify each timestamp into regime
│   │   └── breakdown.py             # Per-regime metrics
│   │
│   ├── statistics/
│   │   ├── __init__.py
│   │   ├── deflated_sharpe.py       # Bailey & Lopez de Prado
│   │   └── bootstrap.py             # Bootstrap confidence intervals
│   │
│   ├── ledger/
│   │   ├── __init__.py
│   │   ├── experiment_ledger.py     # Append-only JSONL
│   │   └── fingerprints.py          # AST fingerprint index
│   │
│   └── sandbox/
│       ├── __init__.py
│       ├── docker_runner.py         # Run factor in isolated container
│       └── restricted_exec.py       # In-process fallback sandbox
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/                        # Fast, no I/O
│   │   ├── test_operators_causality.py
│   │   ├── test_operators_math.py
│   │   ├── test_operators_timeseries.py
│   │   ├── test_ast_similarity.py
│   │   ├── test_complexity.py
│   │   ├── test_deflated_sharpe.py
│   │   ├── test_costs.py
│   │   └── test_regime_tagger.py
│   ├── integration/                 # Uses test fixtures of real data
│   │   ├── test_walk_forward.py
│   │   ├── test_backtest_end_to_end.py
│   │   ├── test_look_ahead_detection.py
│   │   └── test_experiment_ledger.py
│   ├── property/                    # Hypothesis-based property tests
│   │   ├── test_operator_properties.py
│   │   └── test_splits_no_leakage.py
│   └── fixtures/
│       ├── btc_1h_sample.parquet    # Small real data for tests
│       ├── eth_1h_sample.parquet
│       └── funding_sample.parquet
│
├── examples/
│   ├── 01_download_data.ipynb
│   ├── 02_first_strategy.ipynb
│   ├── 03_walk_forward.ipynb
│   └── 04_regime_analysis.ipynb
│
├── scripts/
│   ├── download_all_data.py         # One-command data bootstrap
│   └── build_alpha_zoo.py           # Build the seed strategy library
│
└── docs/
    ├── architecture.md
    ├── operators.md                 # Auto-generated from registry
    ├── api.md
    └── methodology.md               # The "why" of every choice
```

---

## 5. Data Specification

### What to collect in Phase 1

| Dataset | Source | Frequency | History | Format |
|---------|--------|-----------|---------|--------|
| BTC/USDT OHLCV | Binance (ccxt) | 1h | 2017-01-01 to now | parquet |
| ETH/USDT OHLCV | Binance (ccxt) | 1h | 2017-08-01 to now | parquet |
| BTC/USDT OHLCV (daily) | Binance (ccxt) | 1d | 2017-01-01 to now | parquet |
| ETH/USDT OHLCV (daily) | Binance (ccxt) | 1d | 2017-08-01 to now | parquet |
| BTC/USDT:USDT funding | Binance Futures | 8h (forward-fill to 1h) | 2019-09-01 to now | parquet |
| ETH/USDT:USDT funding | Binance Futures | 8h (forward-fill to 1h) | 2019-09-01 to now | parquet |
| BTC Open Interest | Binance Futures | 1h | 2020-01-01 to now | parquet |
| ETH Open Interest | Binance Futures | 1h | 2020-01-01 to now | parquet |
| Fear & Greed Index | Alternative.me | 1d | 2018-02-01 to now | parquet |
| BTC active addresses | Blockchain.com | 1d | 2009 to now | parquet |
| BTC hashrate | Blockchain.com | 1d | 2009 to now | parquet |
| BTC dominance | CoinGecko | 1d | 2013 to now | parquet |
| DXY (dollar index) | yfinance | 1d | 2015 to now | parquet |
| SPY (S&P 500 ETF) | yfinance | 1d | 2015 to now | parquet |
| Total stablecoin mcap | DefiLlama | 1d | 2019 to now | parquet |
| CryptoPanic news | CryptoPanic API | daily batch | 2025 onwards (real-time) | parquet |

**Total disk footprint:** Under 50 MB. Easily fits in git-lfs or just a cloud bucket.

### Data splits (enforced in `data/splits.py`)

```
Train:      2017-01-01 to 2023-12-31  (~7 years)
Validation: 2024-01-01 to 2024-12-31  (~1 year)
Test:       2025-01-01 to today       (~1.3 years, grows over time)
```

**Rules:**
- `load_train()` returns train only
- `load_train_plus_validation()` returns train + val for final model fit
- `load_test()` requires `reveal_test_set=True, reason="<description>"` and is logged
- Mixing splits requires explicit flags

### Data schema (enforced via Pandera)

Every dataset loaded from disk is validated against a Pandera schema. If the schema fails, loading raises `DataSchemaViolation`.

```python
# OHLCV schema
class OHLCVSchema(pa.DataFrameModel):
    timestamp: Series[pd.Timestamp] = pa.Field(unique=True, gt=pd.Timestamp("2009-01-01"))
    open: Series[float] = pa.Field(gt=0)
    high: Series[float] = pa.Field(gt=0)
    low: Series[float] = pa.Field(gt=0)
    close: Series[float] = pa.Field(gt=0)
    volume: Series[float] = pa.Field(ge=0)

    @pa.check("high")
    def high_ge_open_close(cls, series, df): ...

    @pa.check("low")
    def low_le_open_close(cls, series, df): ...
```

---

## 6. The Operator Library — Full Specification

### Core rules

1. Every operator is a pure function.
2. Every operator is registered in the registry at import time.
3. Every operator has a causality certificate (proven not to look ahead).
4. Every operator has a docstring with: description, inputs, output, example.
5. No operator uses `.shift(-n)` or any negative shift.
6. No operator uses `.iloc` with positive forward indexing.

### Full list to implement

**Time-series operators (`operators/timeseries.py`):**

```python
ts_mean(x: Series, window: int) -> Series
ts_std(x: Series, window: int) -> Series
ts_min(x: Series, window: int) -> Series
ts_max(x: Series, window: int) -> Series
ts_rank(x: Series, window: int) -> Series      # rank within window
ts_zscore(x: Series, window: int) -> Series
ts_diff(x: Series, lag: int) -> Series          # x - x.shift(lag), lag>0
ts_pct_change(x: Series, lag: int) -> Series
ts_skew(x: Series, window: int) -> Series
ts_kurt(x: Series, window: int) -> Series
ts_quantile(x: Series, window: int, q: float) -> Series
ts_corr(x: Series, y: Series, window: int) -> Series
ts_cov(x: Series, y: Series, window: int) -> Series
ts_argmax(x: Series, window: int) -> Series    # days-ago of max
ts_argmin(x: Series, window: int) -> Series
ts_decay_linear(x: Series, window: int) -> Series  # weighted avg, linear decay
ts_ema(x: Series, halflife: int) -> Series
```

**Math operators (`operators/math.py`):**

```python
add(a, b), sub(a, b), mul(a, b), div(a, b)
log(x), exp(x), abs(x), sign(x), sqrt(x)
power(x, p), tanh(x), sigmoid(x)
clip(x, lo, hi)
```

**Conditional operators (`operators/conditional.py`):**

```python
if_else(cond: Series, a: Series, b: Series) -> Series
greater_than(x, y), less_than(x, y), equal(x, y)
and_(a, b), or_(a, b), not_(a)
```

**Crypto-specific operators (`operators/crypto.py`):**

```python
funding_z(symbol: str, window: int) -> Series
oi_change(symbol: str, window: int) -> Series
fear_greed(window: int) -> Series          # value from F&G index
btc_dominance_change(window: int) -> Series
stablecoin_mcap_change(window: int) -> Series
active_addresses_change(symbol: str, window: int) -> Series
hashrate_change(window: int) -> Series     # BTC only
dxy_change(window: int) -> Series
spy_correlation(symbol: str, window: int) -> Series
```

---

## 7. The Factor System

### Factor representation

A factor is represented as an AST (Abstract Syntax Tree) that can be:
1. Serialized to JSON (for storage in the ledger)
2. Compiled to a callable (for execution)
3. Compared to other factors (for similarity checks)
4. Displayed as a human-readable formula (for the wiki)

```python
@dataclass(frozen=True)
class FactorNode:
    operator: str                    # e.g., "ts_mean"
    args: tuple                      # tuple of FactorNode | primitives
    kwargs: dict                     # e.g., {"window": 20}

@dataclass(frozen=True)
class Factor:
    name: str                        # human-readable, e.g., "funding_zscore_24h"
    description: str                 # natural language
    hypothesis: str                  # the hypothesis this implements
    root: FactorNode                 # the AST root
    metadata: dict                   # tags, author, timestamp, etc.
```

### AST similarity algorithm

Two factors' similarity is measured by the largest common subtree (tree edit distance alternative):

```python
def ast_similarity(a: FactorNode, b: FactorNode) -> float:
    """Returns similarity in [0, 1]. 1.0 = identical structure."""
```

**Threshold:** If `ast_similarity(new_factor, any_existing_factor) > 0.7`, the factor is rejected as "too similar to existing work."

### Complexity metrics

```python
def factor_complexity(f: Factor) -> dict:
    return {
        "ast_depth": ...,           # max depth of tree
        "node_count": ...,          # total nodes
        "unique_operators": ...,    # distinct operators used
        "unique_features": ...,     # distinct raw data fields
        "total_params": ...,        # free parameter count
    }
```

**Rejection thresholds:**
- `ast_depth > 7` → reject (too nested, likely overfit)
- `node_count > 30` → reject
- `unique_features > 6` → reject (too many data sources)

---

## 8. The Backtest Engine

### Public API (the ONLY entry point)

```python
from crypto_alpha_engine import BacktestEngine

engine = BacktestEngine(
    train_data=load_train(["BTC/USDT", "ETH/USDT"], "1h"),
    cost_model=CostModel(taker_bps=10, maker_bps=2, slippage_model="volume_based"),
    regime_tagger=RegimeTagger(),
    experiment_ledger=ExperimentLedger("./ledger.jsonl"),
)

result: BacktestResult = engine.run(
    factor=my_factor,
    start_date="2019-01-01",
    end_date="2023-12-31",
    rebalance="1h",
    position_sizing="rank_long_short",  # or "long_only", "threshold"
    walk_forward_config=WalkForwardConfig(
        train_months=24,
        test_months=3,
        step_months=1,
    ),
)
```

### What BacktestResult contains (frozen dataclass)

```python
@dataclass(frozen=True)
class BacktestResult:
    # Identity
    factor_id: str                    # hash of factor
    factor_name: str
    run_timestamp: datetime
    data_version: str                 # hash of data used

    # Headline metrics (ALL post-cost)
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float               # as negative number
    annualized_return: float
    total_return: float

    # Predictive power
    ic_mean: float                    # information coefficient
    ic_std: float
    ic_ir: float                      # IC information ratio
    hit_rate: float                   # % of trades profitable

    # Cost breakdown
    gross_sharpe: float               # before costs
    net_sharpe: float                 # after costs (same as sharpe)
    turnover_annual: float
    total_fees_paid: float
    total_slippage_paid: float

    # Regime breakdown (Sharpe per regime)
    bull_sharpe: float
    bear_sharpe: float
    crab_sharpe: float
    high_vol_sharpe: float
    low_vol_sharpe: float
    normal_vol_sharpe: float
    euphoric_funding_sharpe: float
    fearful_funding_sharpe: float
    neutral_funding_sharpe: float

    # Robustness
    walk_forward_sharpe_mean: float
    walk_forward_sharpe_std: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    deflated_sharpe_ratio: float      # corrected for N experiments
    n_experiments_in_ledger: int

    # Factor properties
    factor_ast_depth: int
    factor_node_count: int
    factor_max_similarity_to_zoo: float

    # Trade statistics (aggregate only, NEVER trade-level)
    n_trades: int
    avg_trade_duration_hours: float
    avg_position_size: float
    max_leverage_used: float
```

Note the absence of: equity curve, trade timestamps, per-trade PnL, daily returns series, drawdown dates. These are NEVER returned.

### Walk-forward configuration

```python
@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 24            # rolling train window
    test_months: int = 3              # rolling test window
    step_months: int = 1              # step forward between windows
    min_train_months: int = 12        # minimum before first prediction
```

### Cost model

```python
@dataclass(frozen=True)
class CostModel:
    taker_bps: float = 10.0           # 10 bps = 0.1% per side
    maker_bps: float = 2.0
    slippage_model: str = "volume_based"
    # volume_based: 0.05% if trade < 1% daily volume,
    #               quadratically increases up to 1% at 10% daily volume
    funding_applied: bool = True      # for perp strategies
    borrow_rate_bps: float = 20.0     # for shorts on spot
```

---

## 9. Regime Tagging

### Three independent regime dimensions

Every timestamp is tagged with three labels:

**Trend regime** (based on BTC daily):
- `bull`: BTC close > 200d SMA AND 200d SMA slope > 0 (over 30d)
- `bear`: BTC close < 200d SMA AND 200d SMA slope < 0 (over 30d)
- `crab`: otherwise

**Volatility regime** (based on BTC 30d realized vol, annualized):
- `low_vol`: < 40%
- `normal_vol`: 40-80%
- `high_vol`: > 80%

**Funding regime** (based on BTC 7d avg funding rate):
- `euphoric`: > 0.05% per 8h
- `fearful`: < -0.02% per 8h
- `neutral`: otherwise

**Rule:** Regime labels are computed using only past data (causal). A bar at time `t` can only be labeled using data up to `t`.

---

## 10. Deflated Sharpe Ratio

Implementation based on Bailey & Lopez de Prado (2014). Corrects observed Sharpe for multiple-testing bias.

```python
def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,              # from experiment ledger
    returns_skew: float,
    returns_kurt: float,
    n_observations: int,
    sharpe_variance_across_trials: float,
) -> float:
    """
    Returns the probability-adjusted Sharpe ratio.
    DSR > 0.95 means strategy is statistically significant after correcting
    for the number of trials performed.
    """
```

**Publication threshold:** `deflated_sharpe_ratio > 0.95` AND `out_of_sample_sharpe > 1.0`.

---

## 11. Experiment Ledger

### File format: append-only JSONL

```json
{"timestamp": "2026-04-24T14:32:01Z", "factor_id": "f_a7b3c2", "factor_hash": "...", "ast_depth": 4, "node_count": 12, "sharpe": 1.4, "deflated_sharpe": 0.73, "status": "validated", ...}
```

### Ledger API

```python
class ExperimentLedger:
    def append(self, result: BacktestResult, factor: Factor): ...
    def count_experiments(self) -> int: ...
    def get_fingerprints(self) -> list[FactorFingerprint]: ...
    def get_sharpe_distribution(self) -> np.ndarray: ...
    def get_coverage_by_modality(self) -> dict: ...
    def get_coverage_by_regime(self) -> dict: ...
```

### Never modify, only append
The ledger is opened in append-only mode. An os-level check confirms no existing line was ever modified (by checking file inode + size grows monotonically).

---

## 12. Sandbox (Phase 1.5 — can defer)

For Phase 1, the sandbox is optional. Since factors are composed from the operator library (not raw Python), the risk of malicious code is low. But for production:

### Docker sandbox
- Read-only mount of `data/train/` only
- No network
- Memory limit: 512 MB
- CPU limit: 1 core
- Timeout: 60 seconds per factor

### In-process sandbox (fallback)
- `RestrictedPython` for AST validation
- Operator-registry whitelist enforcement
- AST visitor rejects any non-registered node type

---

## 13. Testing Strategy

### Test pyramid

```
                  /\
                 /  \     Property tests (5%)
                /----\    Integration tests (20%)
               /      \   Unit tests (75%)
              /--------\
```

### Required test coverage: 90% minimum

### Unit tests — MUST PASS before module is merged

#### `test_operators_causality.py`
For EVERY operator in the registry:
- Test 1: Output at index `t` does not change if you modify input at index `t+1`
- Test 2: Output at index `t` is NaN if insufficient history (< window)
- Test 3: Output is deterministic (same input → same output)
- Test 4: Operator raises on negative window

```python
def test_ts_mean_causality():
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range(...))
    result_full = ts_mean(x, window=3)

    # Modify future values
    x_modified = x.copy()
    x_modified.iloc[5:] = 999

    result_modified = ts_mean(x_modified, window=3)

    # Past values must be identical
    pd.testing.assert_series_equal(
        result_full.iloc[:5],
        result_modified.iloc[:5]
    )
```

#### `test_ast_similarity.py`
- Test: identical ASTs have similarity 1.0
- Test: completely different ASTs have similarity < 0.3
- Test: ASTs differing only in constants have high similarity (>0.8)
- Test: similarity is symmetric: `sim(a, b) == sim(b, a)`
- Test: similarity is in [0, 1] for 100 random AST pairs

#### `test_complexity.py`
- Test: depth calculation correct for hand-built ASTs of depth 1, 3, 7
- Test: node count correct
- Test: complexity rejection threshold fires at depth 8

#### `test_deflated_sharpe.py`
- Test: DSR decreases as n_trials increases (all else equal)
- Test: DSR = observed Sharpe when n_trials = 1, skew = 0, kurt = 3
- Test: DSR handles edge cases (n_trials = 0, observed_sharpe = 0)
- Test: reference implementation matches published paper's worked example

#### `test_costs.py`
- Test: fees applied symmetrically on buy and sell
- Test: slippage increases with position size / volume ratio
- Test: funding applied only to perp positions
- Test: costs cannot be set to zero (raises ConfigError)

#### `test_regime_tagger.py`
- Test: regime labels use only past data (causal)
- Test: all three regime dimensions return valid labels for every timestamp
- Test: bull + bear + crab partition (every timestamp has exactly one trend label)
- Test: regime boundaries match hand-computed examples

### Integration tests — MUST PASS before engine release

#### `test_look_ahead_detection.py`
The canary test. Deliberately construct a factor with look-ahead bias. The engine MUST catch it:

```python
def test_look_ahead_is_rejected():
    # Factor that (maliciously) uses future returns
    def cheating_factor(data):
        return data["close"].shift(-1) / data["close"] - 1  # LOOKS FORWARD

    with pytest.raises(LookAheadDetected):
        engine.run(cheating_factor, ...)
```

The engine detects this by: re-running the factor with shifted data and checking if outputs at past indices change.

#### `test_walk_forward.py`
- Test: walk-forward produces non-overlapping test windows
- Test: each test window uses only data from prior train window
- Test: aggregating walk-forward results gives correct overall Sharpe
- Test: minimum training data is enforced

#### `test_backtest_end_to_end.py`
- Test: simple momentum strategy produces reasonable Sharpe on BTC 2020-2023
- Test: buy-and-hold produces expected return matching BTC's actual return
- Test: all BacktestResult fields are populated (no None)
- Test: gross_sharpe > net_sharpe (costs have non-zero effect)

#### `test_experiment_ledger.py`
- Test: appending preserves all previous lines byte-for-byte
- Test: ledger cannot be modified externally (corruption detected)
- Test: counting across 1000 appends is accurate

### Property tests (using `hypothesis`)

#### `test_operator_properties.py`
- Property: ts_mean followed by ts_mean = ts_mean with wider window (approximately)
- Property: z-score of constant series is NaN or 0
- Property: for all series x, ts_min(x, w) <= ts_max(x, w)
- Property: add is commutative: add(x, y) == add(y, x)

#### `test_splits_no_leakage.py`
- Property: for any random sample from train, no timestamp is in validation set
- Property: for any random sample from test, no timestamp is in train set
- Property: max timestamp in train < min timestamp in validation

### Test fixtures
Use 30 days of real BTC/ETH data (small parquet files ~100KB) as fixtures. NOT synthetic data — tests should catch real-world issues.

### CI/CD (GitHub Actions)
- On every push: run unit tests + integration tests
- On PR: run property tests (slower)
- Block merge if coverage drops below 90%
- Block merge if any test fails

---

## 14. Implementation Order (for the AI coding agent)

**Follow this order strictly. Each phase must fully pass tests before moving to the next.**

### Phase 1: Foundation (Day 1-2)
1. Set up repo structure, pyproject.toml, CI
2. Implement `types.py` (all frozen dataclasses)
3. Implement `exceptions.py`
4. Write `conftest.py` with shared fixtures

**Exit criteria:** `pytest tests/` runs (even if empty), CI passes.

### Phase 2: Data Layer (Day 3-4)
1. Implement `data/schemas.py` (Pandera schemas)
2. Implement `data/downloader.py` (Binance OHLCV + funding)
3. Implement `data/free_sources.py` (F&G, Blockchain.com, DefiLlama, yfinance)
4. Implement `data/splits.py` (train/val/test enforcement)
5. Implement `data/loader.py`

**Exit criteria:** `scripts/download_all_data.py` successfully pulls all data. Loading any split returns valid, schema-validated DataFrames. All `test_splits_no_leakage.py` property tests pass.

### Phase 3: Operator Library (Day 5-6)
1. Implement `operators/registry.py`
2. Implement `operators/timeseries.py` (all ts_* operators)
3. Implement `operators/math.py`
4. Implement `operators/conditional.py`
5. Implement `operators/crypto.py` (requires data layer)
6. Write `operators/causality_check.py`

**Exit criteria:** All `test_operators_*.py` tests pass. `registry.list_operators()` returns 40+ operators. Causality test passes for every single operator.

### Phase 4: Factor System (Day 7-8)
1. Implement `factor/ast.py` (FactorNode, Factor)
2. Implement `factor/parser.py` (JSON → AST)
3. Implement `factor/compiler.py` (AST → callable)
4. Implement `factor/similarity.py` (AST similarity)
5. Implement `factor/complexity.py`

**Exit criteria:** Can serialize/deserialize factors. Can compile factor and run it on data. All AST similarity and complexity tests pass.

### Phase 5: Metrics & Regimes (Day 9)
1. Implement `backtest/metrics.py` (Sharpe, Sortino, IC, etc.)
2. Implement `regime/tagger.py`
3. Implement `regime/breakdown.py`
4. Implement `statistics/deflated_sharpe.py`

**Exit criteria:** All metric tests pass against known benchmarks (e.g., SPY's long-term Sharpe is a known quantity).

### Phase 6: Backtest Engine (Day 10-12)
1. Implement `backtest/costs.py`
2. Implement `backtest/position_sizing.py`
3. Implement `backtest/walk_forward.py`
4. Implement `backtest/engine.py`

**Exit criteria:** `test_backtest_end_to_end.py` passes. Can run a simple factor and get a valid BacktestResult. Look-ahead detection works.

### Phase 7: Ledger (Day 13)
1. Implement `ledger/experiment_ledger.py`
2. Implement `ledger/fingerprints.py`
3. Wire ledger into engine

**Exit criteria:** Running 100 backtests appends 100 valid entries. Coverage queries work.

### Phase 8: Polish & Docs (Day 14-15)
1. Write example notebooks
2. Generate operator docs from registry
3. Write README with quickstart
4. Write methodology.md
5. Run full test suite, ensure 90% coverage

**Exit criteria:** A new user can `git clone`, `pip install -e .`, run the quickstart notebook, and get a valid backtest in under 5 minutes.

---

## 15. Style & Code Quality

### Code style
- **Formatter:** `black` (line length 100)
- **Linter:** `ruff`
- **Type checker:** `mypy --strict` (yes, strict mode)
- **Docstrings:** Google style, REQUIRED on every public function

### Commit hygiene
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- Each commit passes tests on its own
- No commit > 300 lines changed without a design rationale in the message

### Type hints
- Every function signature is fully typed
- `pd.Series` is the standard type for time series
- `pd.DataFrame` is the standard type for multi-column data
- Use `typing.Protocol` for interfaces, not ABCs

### Logging
- Use `structlog` for structured logging
- Every backtest logs entry + exit with factor ID, timestamp, result summary
- NEVER log trade-level detail (this would leak data through log files)

### Error messages
- Every exception message includes: what went wrong, what the function expected, what was received
- Example: `raise LookAheadDetected(f"Factor {factor.name} uses future data at index {t}: expected causal, got forward-looking reference")`

---

## 16. Documentation Requirements

### README.md must include:
- 30-second pitch (what this is, why it exists)
- Quickstart (5 lines of code to first backtest)
- The 12 quant mistakes we prevent (with links to where each is prevented in code)
- Link to methodology.md
- How to contribute
- License

### methodology.md must include:
- Why walk-forward only (citations)
- Why sealed engine (threat model: what can the user/AI do wrong?)
- Cost model justification (why 10 bps? why volume-based slippage?)
- Regime definitions and their rationale
- Deflated Sharpe explanation (with worked example)

### API docs (auto-generated with mkdocs)
- Every public class and function
- Example for every operator (auto-extracted from docstring)

---

## 17. What NOT to build in Phase 1

To scope down and ship, explicitly defer:
- ❌ Evolutionary mutation/crossover (Phase 2 in the lab repo)
- ❌ AI agent integration (belongs in `crypto-alpha-lab`, not engine)
- ❌ Web UI (Phase 4)
- ❌ Docker sandbox (nice-to-have; factors from operator library are already safe)
- ❌ Multi-asset universe beyond BTC+ETH (Phase 5)
- ❌ Live trading (out of scope forever — this is research)
- ❌ Portfolio optimization (single-factor per strategy for now)

---

## 18. Definition of Done for Phase 1

The engine is considered complete when:

1. ✅ All tests pass (unit + integration + property)
2. ✅ Test coverage ≥ 90%
3. ✅ `mypy --strict` passes with zero errors
4. ✅ `ruff check` passes with zero warnings
5. ✅ README quickstart actually works in a clean environment
6. ✅ A hand-coded momentum strategy on BTC 2020-2023 produces a valid BacktestResult
7. ✅ A deliberately cheating strategy is rejected with `LookAheadDetected`
8. ✅ Operator registry has at least 40 registered operators
9. ✅ Experiment ledger correctly tracks 100+ appended experiments
10. ✅ The methodology.md explains every design decision

---

## 19. Key libraries & versions

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=15.0",           # parquet
    "ccxt>=4.3",               # exchange data
    "pandera>=0.19",           # schema validation
    "vectorbt>=0.26",          # fast backtesting primitives
    "scipy>=1.12",             # stats
    "structlog>=24.1",         # logging
    "yfinance>=0.2.40",        # macro data
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "hypothesis>=6.100",       # property testing
    "mypy>=1.9",
    "ruff>=0.3",
    "black>=24.3",
    "mkdocs>=1.5",
    "mkdocs-material>=9.5",
]
```

---

## 20. Final instructions for the AI coding agent

When implementing this spec:

1. **Read the whole spec first.** Do not start typing until you understand section 14 (implementation order) and section 13 (testing strategy).

2. **Write tests first.** For each module, write the failing tests listed in section 13, then implement the module to make them pass. This is TDD, non-negotiable.

3. **Do not invent features.** If something is not in this spec, it's out of scope. Propose additions via an issue, don't merge them silently.

4. **Do not compromise on causality.** If a test fails because an operator uses future data, do not modify the test. Fix the operator.

5. **Ask when uncertain.** If a section is ambiguous, raise it rather than guess. Cost of clarification: minutes. Cost of wrong assumption: days.

6. **Keep commits small.** Each commit should be a coherent, testable unit of work.

7. **Update docs as you go.** Don't leave docs for the end. Docstrings and methodology notes are part of the feature.

8. **The engine is a library, not a script.** Every public API should be importable and usable standalone. No global state, no side effects at import time.

9. **Paranoia about data leakage is a feature.** Err on the side of rejecting a factor rather than letting a suspicious one through. The engine's value IS its strictness.

10. **When in doubt, look at the methodology.** Every design choice has a justification in methodology.md. If you're unsure why something is structured a way, it's likely written down.

---

*End of specification.*

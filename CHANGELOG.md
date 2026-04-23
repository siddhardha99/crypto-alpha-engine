# Changelog

All notable changes to this project will be documented in this file.
The format is based on [keepachangelog.com](https://keepachangelog.com);
this project adheres to [semantic versioning](https://semver.org).

## [1.0.0] - 2026-04-23

First public release. Sealed, cheat-proof backtesting engine for
crypto quantitative research. Eight build phases (SPEC §14) each
landed behind a structural invariant — the invariants are now
enforced by the code and tests, not by convention.

### Added

- **Core types and exception hierarchy** — frozen dataclasses for
  every engine data contract, typed exception tree with
  `CryptoAlphaEngineError` at the root (SPEC §15, §16).
- **Data layer** — Coinbase spot OHLCV, BitMEX perp funding, plus six
  free-tier macro / on-chain / sentiment sources behind the
  `DataSource` Protocol (SPEC §5). Pandera-validated ingestion,
  idempotent downloads, quarantine-on-schema-violation for
  corrupted files, three-way train / validation / test splits.
- **Operator library** — 46 causal primitives across four
  categories (timeseries, math, conditional, crypto). Each operator
  ships with a per-operator causality test plus an entry in the
  registry-walking coverage canary; the canary fires if a new
  operator is registered without a catalogue entry (SPEC §6, §13).
- **Factor system** — `FactorNode` AST, allowlist-based parser
  (zero `eval` / `exec` / `compile` paths), two-stage compiler with
  subtree memoisation, `ast_similarity` + `behavioural_similarity`
  for duplicate detection, four-component `factor_complexity` score
  (SPEC §7). Vocabulary-closure canary: no operator may take a
  literal string argument.
- **Metrics and regimes** — 16 financial metrics (Sharpe, Sortino,
  Calmar, IC mean/std/IR, profit factor, hit rate, etc.) with the
  Phase-5 NaN convention on degenerate input. Bailey & Lopez de
  Prado (2014) Deflated Sharpe Ratio, validated against the paper's
  Section IV worked example to ±1%. Causal regime tagging for
  trend / volatility / funding dimensions, each with an explicit
  future-perturbation causality test (SPEC §8, §9, §10).
- **Backtest engine** — walk-forward orchestration with a three-
  ordering fold-independence canary. Vectorbt kept behind a single
  boundary (`backtest/simulation.py`) enforced by a grep-based
  import canary. Two-layer causality checks on every factor:
  Layer 1 (AST whitelist) + Layer 2 (runtime perturbation).
  `run_backtest` returns a fully populated `BacktestResult` across
  ~35 fields, no trade-level time series ever exposed (SPEC §8).
- **Experiment ledger** — append-only JSONL with per-line schema
  versioning. Sentinel encoding for `inf` / `-inf` / `nan` floats
  restricted to known numeric fields, with `json.dumps(allow_nan=False)`
  as defensive trap. SPEC §7 duplicate detection (structural 0.7 /
  behavioral 0.9 thresholds) with bounded work: descending-similarity
  iteration, newest-first tie-break, 20-match hard cap, and
  explicit `DuplicateCheckSaturated` on exceedance. DSR inputs
  (n_trials + V[SR]) auto-wire from ledger statistics (SPEC §11).
- **Documentation** — [SPEC.md](SPEC.md) (build specification),
  [docs/methodology.md](docs/methodology.md) (design rationale),
  [docs/factor_design.md](docs/factor_design.md) (factor contract),
  [docs/adding_custom_sources.md](docs/adding_custom_sources.md)
  (DataSource template), [CONTRIBUTING.md](CONTRIBUTING.md) (PR
  expectations), [examples/research_workflow.py](examples/research_workflow.py)
  (worked research loop on real BTC data), and a two-tier
  quickstart in the README.

### Not included in v1.0

Each deferred deliberately. One-line reasons so future GitHub issues
can be closed with a link:

- **Aggregated open interest data** — no free-tier provider with
  ≥2-year history exists in 2026; contributors implement OI via the
  `DataSource` Protocol until that changes.
- **Live trading / execution** — this is a research engine; live
  execution is a different security model and scope.
- **AI-powered factor generation** — `DataSource` and operator
  registry architectures are designed to support agent callers, but
  no agent ships in v1.0.
- **PyPI distribution** — install from GitHub for now; PyPI
  publication is a v1.1 candidate once there is a user base to
  justify the release-workflow overhead.
- **Multi-asset universes beyond BTC/ETH** — Phase-1 scope narrowed
  to single-asset to keep the engine auditable; multi-asset
  composition is a future-version item.
- **Sub-monthly `WalkForwardConfig`** — SPEC §8 is explicit about
  calendar-month semantics; sub-monthly windows would require a
  re-specification and are deferred.

### Known limitations

- **Test fixtures are one month of real data** — the committed
  `tests/fixtures/*.parquet` slice is January 2022, small enough to
  fit in the repo. Full walk-forward backtests need multi-year data
  pulled via `scripts/download_all_data.py` (~45s one-time download).
- **Cost model uses a flat slippage floor during simulation.** The
  volume-based quadratic model lives in `backtest/costs.py` for
  explicit use, but vectorbt's simulation receives a scalar floor
  because trade notional isn't known at sim-entry time. Phase 7+
  enhancement.
- **`_max_int_heuristic` in duplicate detection doesn't compose.**
  A factor like `ts_mean(ts_mean(x, 20), 30)` has true warmup
  20 + 30 − 1 = 49 bars; the heuristic returns 30. The 3× multiplier
  in `check_duplicate`'s data-sufficiency check buys slack for
  moderate nesting but can undersize for pathological depth.
  Workaround: pass longer feature history.
- **Single-factor backtests only.** Portfolio-of-factors composition
  — weight allocation across factors, cross-factor correlation
  penalties — is caller-side for v1.0.

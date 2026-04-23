# Methodology

Design decisions and their rationale. Read this when you want to
understand a non-obvious choice, or when you're considering changing
one.

## Table of contents

- [Data sources: US-reachable only](#data-sources-us-reachable-only)
- [Funding rates: signals, not PnL](#funding-rates-signals-not-pnl)
- [Extensibility: sources as plugins](#extensibility-sources-as-plugins)
- [The AST vocabulary layer: when to add a "crypto" operator](#the-ast-vocabulary-layer-when-to-add-a-crypto-operator)
- [What's deliberately not built-in](#whats-deliberately-not-built-in)
- [Paper-example tests: external validation over self-consistency](#paper-example-tests-external-validation-over-self-consistency)
- [Why two causality layers](#why-two-causality-layers)

*(More sections will land as later phases are implemented.)*

## Data sources: US-reachable only

Every built-in source in Phase 2 is globally reachable without a VPN,
with at most a free-tier API key. Three major crypto venues are
**deliberately excluded**:

* **Binance** — HTTP 451 from US IPs per their Terms of Service. This
  is not a throttle or rate limit; it's a legal block at the CDN layer.
  US developers and any CI runner in a US datacenter cannot use
  Binance.
* **Bybit** — CloudFront 403 from multiple regions (confirmed from our
  Phase-2 development host). Bybit US exists as a separate entity with
  a narrower product set and API surface, not a drop-in substitute.
* **Binance.US** — lists a small subset of pairs with shallow history
  compared to Binance global. Insufficient for multi-year backtests.

The alternative set — **Coinbase** (spot), **BitMEX** (perp funding),
plus the globally-reachable free sources (alternative.me,
blockchain.com, DefiLlama, yfinance, CoinGecko, CryptoPanic) — works
from every region we've tested, with or without free-tier API keys.
The cost of this choice: the BTC/USD (Coinbase) ↔ BTC/USDT (Binance)
basis is under 5 bps at 1h bars, which is noise relative to 10 bps
taker fees and realistic slippage. For factor research at hourly
timescales, the two are interchangeable.

### Criteria for adding a new exchange

When evaluating a new exchange or data source for inclusion, require:

1. **Reachable from US IPs without a VPN.** Test from a US-based CI
   runner or equivalent, not a local laptop with personal network
   shaping.
2. **Public historical API** (no key, or free-tier key). Paid-tier
   dependencies belong in private forks, not the open-source engine.
3. **History depth ≥ 2 years** for the data type in question.
4. **UTC-timestamped bars or records.** Localized or ambiguous
   timestamp formats are rejected at the schema layer.
5. **Sufficient historical depth for the finest intended timeframe
   (at least 2 years at 1h).** Some exchanges serve deep daily history
   but only a rolling short window (days to weeks) of hourly bars.
   This trap is why Kraken isn't our OHLCV fallback despite being
   reachable — Kraken's 1h OHLCV is a rolling ~30-day window.

## Funding rates: signals, not PnL

BitMEX perps are BTC-margined (inverse), not USDT-margined (linear). A
factor that takes a long BTC perp position on BitMEX accrues PnL in
BTC; on Binance it accrues in USDT. This distinction matters for
trade-level accounting, but our use of funding rates is as a
**signal** — "what is the market paying for leverage in this
direction?" — not as a PnL source. The funding rate itself (fraction
paid per period) is the same concept across inverse and linear perps,
and moves together on any given day with tight cross-exchange
correlation. Our factor operators treat funding values as
dimensionless feature inputs; the backtest PnL comes from spot.

If a future strategy explicitly *trades* perps rather than treating
funding as a signal, the inverse/linear distinction becomes a
first-class engine concern and this section gets revisited.

## Extensibility: sources as plugins

The `DataSource` Protocol and registry (SPEC §5.1) exist for two
reasons:

1. **Private alpha sources.** Most serious quant research uses data
   that isn't free — proprietary order-book feeds, vendor sentiment,
   Kaiko-licensed tick data, Amberdata on-chain. Keeping the engine
   open-source while supporting private data requires a sharp seam
   between the two: the engine owns schemas, the source owns the
   fetch. Contributors can wire in a private source without forking
   the repo.

2. **Community contributions.** A contributor who trusts a specific
   exchange or aggregator that isn't in our default set is one class
   + one test away from adding it. No engine code change, no schema
   change, no new dependency on their vendor.

The alternative — hardcoding the exchange list — makes every decision
like "switch from Binance to Coinbase" a migration. The Protocol makes
it a new file.

## The AST vocabulary layer: when to add a "crypto" operator

Every operator in `crypto_alpha_engine/operators/crypto.py` is a **pure
renaming alias** over a generic timeseries kernel:

* `funding_z == ts_zscore`
* `fear_greed == ts_mean`
* `oi_change`, `btc_dominance_change`, `stablecoin_mcap_change`,
  `active_addresses_change`, `hashrate_change`, `dxy_change` all
  `== ts_pct_change`
* `spy_correlation == ts_corr`

No default windows, no sign conventions, no bounds checks — if you read
the source, each wrapper is one line of delegation plus a docstring.

**Why they exist anyway:** factor ASTs are a user-authored, ledger-
stored artefact. `FactorNode("funding_z", args=("BTC/USD:BTC", 24))`
captures intent — *"z-score the funding rate"* — in a way that
`FactorNode("ts_zscore", args=("BTC/USD:BTC_funding", 24))` does not.
The AST is what a human reads in the experiment ledger, what an AI
agent composes, and what our similarity / complexity metrics see. A
rich domain vocabulary at that layer pays off in every downstream
consumer.

**When to add a new crypto-layer operator:**

1. It has a domain name that factor authors naturally reach for.
2. The computation maps cleanly to an existing timeseries kernel. If
   it needs genuinely new math (not in `timeseries.py`), add the math
   to `timeseries.py` first and then wrap.
3. It binds to a specific data semantic (funding, OI, sentiment, etc.)
   rather than a generic "take these two columns and correlate them".

**When to SKIP the crypto wrapper and just use the generic operator:**

1. The relationship between inputs isn't specific to a crypto dataset
   (e.g. correlating two arbitrary features — use `ts_corr`, not a
   bespoke wrapper).
2. The operator would differ from the underlying kernel only by a
   default parameter value. Defaults belong in the factor spec, not in
   an operator clone.

This pattern keeps the `timeseries` / `math` / `conditional` namespaces
focused on pure mathematical primitives and lets `crypto` evolve as a
semantic layer that can be extended without modifying the math.

## What's deliberately not built-in

The engine ships a small set of built-in sources. Several obvious
candidates are deliberately **not** shipped — if you want them, they
belong in a contributed DataSource, not in core.

**Aggregated open interest (no built-in source).** Open interest is a
genuinely useful feature — the funding/OI divergence signal is a
standard ingredient in mean-reversion and trend-following factors.
But every aggregated-OI provider we evaluated (Coinglass, CryptoQuant,
Glassnode) charges monthly subscriptions in 2026 that break our
free-tier principle (§4-ish in "Criteria for adding a new exchange"),
and the cheap paid tiers cap history at 180 days — too shallow for
multi-year research anyway.

The honest move is to admit that **aggregated OI is out of scope for
v1.0** rather than silently degrade to a stub or ship a private-key-
dependent integration. Contributors who need OI now can implement it
via the DataSource Protocol, aggregating from free exchange APIs
(BitMEX, OKX, Deribit, Coinbase perps, dYdX, Hyperliquid) themselves.
When such a pipeline matures as an external package it will be
documented in the README as the recommended OI source.

This is the shape of decision the Protocol was designed to support:
**defer-and-extend beats bake-in-a-paid-dependency.** The same rule
applies to any future feature that would require a commercial
subscription — including dedicated sentiment APIs, proprietary
orderbook feeds, and licensed tick data.

## Paper-example tests: external validation over self-consistency

For any formula taken from a published paper, the test suite must
include a **paper-example test** that compares our implementation's
output against the paper's own worked numbers, to within ~1%
tolerance. This rule distinguishes two separate claims:

- *Self-consistency.* Our code matches our own arithmetic. A
  hand-walkthrough test computes the formula in the test file and
  asserts the implementation matches. If we transcribed the formula
  wrong from the paper, both the test and the implementation share
  the error — the test passes, and we've shipped a bug that looks
  correct.
- *External correctness.* Our code matches the paper's published
  output. This requires numbers the paper prints — either in a
  worked example, a table of values, or a figure with axis-readable
  points. Matching those numbers proves the transcription is right.

The canonical instance is `deflated_sharpe.py` against Bailey & Lopez
de Prado (2014). The hand-walkthrough test pinned that our DSR matches
our own arithmetic; the paper-example test (which collapses DSR to
PSR by setting `sharpe_variance_across_trials=0`, then plugs in
Section IV's worked example of n=24, sr=0.459, γ₃=-2.448, γ₄=13.164
and asserts the output matches the paper's 0.9051 to ±0.01) pinned
that we transcribed BLP 2014 correctly. The two tests catch
different bug classes and both are load-bearing.

If the paper doesn't publish worked numbers at all, state that
explicitly in the test docstring and fall back to the strongest
independent check available (e.g., compute the same formula from a
second reference's formulation and compare). Silence on this point
lets transcription errors ship.

## Why two causality layers

`crypto_alpha_engine/backtest/engine.py` runs two independent
causality checks on every factor that enters the engine. They catch
different bug classes, which is why we keep both:

**Layer 1 — AST whitelist.** Every operator registered via
`@register_operator(..., causal_safe=...)` carries a declarative flag
describing whether it respects causality (output at time `t` depends
only on inputs at times `≤ t`). Layer 1 walks the factor AST pre-order
and rejects any factor referencing an operator with `causal_safe=False`.
Static, cheap, runs before any compilation or simulation. Its primary
job is to catch **intentional misregistration** — an operator that
was honestly labeled as acausal by its author reaching a production
backtest by accident.

**Layer 2 — runtime perturbation.** After compilation, the engine
runs the factor at several random cutoffs, perturbs the feature
values at indices ≥ cutoff, and asserts the factor's output at
indices < cutoff is byte-identical to the baseline. If future data
can change past output, the factor is not causal. The test is
behavioral, not declarative: it ignores the `causal_safe` flag and
checks the ground truth.

The justification for Layer 2 is the class of bugs Layer 1 cannot
see:

- **Lying annotations.** An operator kernel does `x.shift(-1)` but
  its `@register_operator` call says `causal_safe=True`. Layer 1's
  whitelist reads the declaration and waves it through. Layer 2's
  perturbation catches the lie.
- **Mistaken annotations.** An operator author writes
  `x.rolling(n, center=True).mean()` — a centered window that peeks
  ahead symmetrically — and genuinely believes it's causal because
  no `.shift(-N)` appears in the code. Layer 1 trusts them. Layer 2
  doesn't.
- **Composition errors.** Two individually-causal operators combine
  in a way that leaks future data through their composition — via
  NaN propagation followed by a fill that consults future values, or
  through an intermediate reindex that shifts output alignment. No
  single operator registration can describe this; the leak only
  exists in the composed factor. Layer 2's behavioral check catches
  it because the composed factor fails the perturbation test, even
  though each operator individually would not.

Layer 1 is cheap and catches intentional violations at submission
time. Layer 2 is more expensive (five extra factor evaluations per
run) and catches the mistakes, lies, and emergent leaks Layer 1
can't see. Running both is the practical cost of a non-negotiable
principle: **causality is sacred, and trust-but-verify is how you
keep it sacred when the people declaring trust are human.**

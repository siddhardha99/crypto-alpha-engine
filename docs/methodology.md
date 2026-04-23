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

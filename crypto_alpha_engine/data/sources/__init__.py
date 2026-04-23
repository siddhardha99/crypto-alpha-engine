"""Built-in data sources — imports auto-register each source.

Importing this package registers every built-in source with
:mod:`crypto_alpha_engine.data.registry`. External (custom) sources
should NOT import from this package; they implement
:class:`crypto_alpha_engine.data.protocol.DataSource` directly and call
``register_source`` themselves. See ``docs/adding_custom_sources.md``.

Registration policy
-------------------

* Sources that need no credentials register unconditionally.
* Sources that require a free-tier API key only register if the env
  var is set at import time. If missing, the source is silently absent
  from the registry — scripts iterating the registry skip it cleanly.
  This is the "gated source" pattern from SPEC §5.1.

Phase 2 built-in sources:

* ``coinbase_spot`` — OHLCV (BTC/USD, ETH/USD).
* ``bitmex_perp`` — FUNDING (BTC/USD:BTC, ETH/USD:BTC).
* ``alternative_fear_greed`` — SENTIMENT.
* ``blockchain_com`` — ONCHAIN (active addresses, hashrate).
* ``defillama_stablecoin`` — MACRO (stablecoin market cap).
* ``coingecko_dominance`` — MACRO (BTC dominance, current-snapshot only).
* ``yfinance_macro`` — MACRO (DXY, SPY).
* ``cryptopanic_news`` — SENTIMENT (gated by ``CRYPTOPANIC_API_KEY``).

No built-in open-interest source ships in Phase 1 — see
``docs/methodology.md`` section "What's deliberately not built-in".
"""

from __future__ import annotations

import os

from crypto_alpha_engine.data.registry import register_source
from crypto_alpha_engine.data.sources.alternative_fear_greed import (
    AlternativeFearGreedSource,
)
from crypto_alpha_engine.data.sources.bitmex_perp import BitmexPerpFundingSource
from crypto_alpha_engine.data.sources.blockchain_com import BlockchainComSource
from crypto_alpha_engine.data.sources.coinbase_spot import CoinbaseSpotSource
from crypto_alpha_engine.data.sources.coingecko_dominance import CoinGeckoDominanceSource
from crypto_alpha_engine.data.sources.defillama_stablecoin import DefiLlamaStablecoinSource
from crypto_alpha_engine.data.sources.yfinance_macro import YFinanceMacroSource

register_source(CoinbaseSpotSource())
register_source(BitmexPerpFundingSource())
register_source(AlternativeFearGreedSource())
register_source(BlockchainComSource())
register_source(DefiLlamaStablecoinSource())
register_source(CoinGeckoDominanceSource())
register_source(YFinanceMacroSource())

# Gated sources: register only when credentials are present.
_CRYPTOPANIC_KEY = os.environ.get("CRYPTOPANIC_API_KEY")
if _CRYPTOPANIC_KEY:
    from crypto_alpha_engine.data.sources.cryptopanic_news import CryptoPanicNewsSource

    register_source(CryptoPanicNewsSource(api_key=_CRYPTOPANIC_KEY))

"""Built-in data sources — imports auto-register each source.

Importing this package registers every built-in source with the
:mod:`crypto_alpha_engine.data.registry`. External (custom) sources
should NOT import from this package; they subclass/implement
:class:`crypto_alpha_engine.data.protocol.DataSource` directly and call
``register_source`` themselves.

Phase 2 built-in sources:

* ``coinbase_spot`` — OHLCV (BTC/USD, ETH/USD).
* (more sources land in Phase 2 closeout — bitmex_perp, coinglass_oi,
  alternative_fear_greed, blockchain_com, defillama_stablecoin,
  coingecko_dominance, yfinance_macro, cryptopanic_news.)
"""

from __future__ import annotations

from crypto_alpha_engine.data.registry import register_source
from crypto_alpha_engine.data.sources.coinbase_spot import CoinbaseSpotSource

# Auto-registration. Idempotent at the registry level (duplicate names
# raise ConfigError), so re-importing the module in the same process
# will trip the duplicate-name guard — that's intentional, and why
# tests use _reset_for_tests() rather than re-importing.
register_source(CoinbaseSpotSource())

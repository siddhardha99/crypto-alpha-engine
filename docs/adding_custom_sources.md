# Adding a custom data source

The engine knows about **data types** (OHLCV, FUNDING, OPEN_INTEREST,
ONCHAIN, SENTIMENT, MACRO), not about specific exchanges or APIs. To
add a new data source — a proprietary vendor, a niche exchange, your
own on-chain indexer — you implement the `DataSource` Protocol and
register it. The engine picks it up through the same path it uses for
the built-in sources.

## The Protocol

```python
from typing import Protocol
import pandas as pd
from crypto_alpha_engine.data.protocol import DataType

class DataSource(Protocol):
    name: str                  # unique identifier across all sources
    data_type: DataType        # OHLCV, FUNDING, ONCHAIN, ...
    symbols: list[str]         # symbols this source supports
    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame: ...
    def earliest_available(self, symbol: str) -> pd.Timestamp: ...
```

Your `fetch()` must return a DataFrame matching the **canonical schema
for its `data_type`** (see `crypto_alpha_engine.data.schemas`). The
engine validates before writing; non-conforming frames go to
`data/quarantine/`.

## A worked example: proprietary sentiment API

Say you have an internal REST API that scores crypto tweets 0.0..1.0
and you want to feed its output into the engine as a sentiment signal.
A complete source in ~30 lines:

```python
# my_pkg/sources/mytwitter.py
from datetime import datetime
import pandas as pd
import requests
from crypto_alpha_engine.data.protocol import DataSource, DataType
from crypto_alpha_engine.data.registry import register_source

class MyTwitterSentiment:
    name = "mytwitter_v1"
    data_type = DataType.SENTIMENT
    symbols = ["BTC", "ETH"]

    def __init__(self, api_key: str, base_url: str = "https://sentiment.internal"):
        self._key = api_key
        self._base = base_url

    def fetch(self, symbol, *, start, end=None, freq="1h"):
        params = {
            "symbol": symbol,
            "since": int(start.timestamp() * 1000),
            "until": int(end.timestamp() * 1000) if end is not None else None,
            "interval": freq,
        }
        resp = requests.get(
            f"{self._base}/scores",
            params=params,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()["scores"]
        return pd.DataFrame({
            "timestamp": pd.to_datetime(
                [r["ts"] for r in rows], unit="ms", utc=True
            ).astype("datetime64[us, UTC]"),
            "value": [float(r["score"]) for r in rows],
        })

    def earliest_available(self, symbol):
        return pd.Timestamp("2022-01-01", tz="UTC")

# Register at import time.
register_source(MyTwitterSentiment(api_key=os.environ["MYTWITTER_KEY"]))
```

## Using your source

Once registered, the loader finds it via the registry:

```python
from crypto_alpha_engine.data.loader import load_sentiment
df = load_sentiment("BTC", source="mytwitter_v1")
```

Or let the registry pick the default for `(DataType.SENTIMENT, "BTC")`:

```python
df = load_sentiment("BTC")     # uses first-registered SENTIMENT source
```

## Checklist before registering

- [ ] `name` is unique across every source the engine will see at runtime.
- [ ] `data_type` matches one of the enum values; don't invent new ones
      without coordinating schema changes.
- [ ] `fetch()` returns a DataFrame with the columns + dtypes required
      by the canonical schema for your `data_type`.
- [ ] Timestamps are UTC-aware, `datetime64[us, UTC]`.
- [ ] Pagination, retries, auth — all handled inside `fetch()`.
- [ ] `earliest_available()` returns a plausible lower bound so the
      orchestrator doesn't ask for data that can't exist.
- [ ] Tests: at minimum, a happy-path test + one failure mode
      (HTTP error, malformed response).

That's it — no engine changes needed.

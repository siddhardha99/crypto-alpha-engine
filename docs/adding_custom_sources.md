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
for its `data_type`** (defined in
`crypto_alpha_engine.data.schemas`, mapped in
`crypto_alpha_engine.data.protocol.CANONICAL_SCHEMAS`). The engine
validates before writing; non-conforming frames go to
`data/quarantine/` and `DataSchemaViolation` is raised.

## Idempotency: the orchestrator handles it, your source doesn't

This is the biggest gotcha for custom-source authors, so read it before
you start coding:

**Your `fetch()` does not need to deduplicate, cache, or track state
between calls.** The engine's download orchestrator
(`run_source` in `crypto_alpha_engine.data.downloader`) reads the
existing parquet for the requested ``(source, symbol, freq)``, computes
the next timestamp to ask for, passes that as `start`, merges the result
with what's on disk, and drops duplicates by `timestamp`. All you do is
honestly return whatever your API gives for ``[start, end)``. Return
overlap if your API forces it — the orchestrator dedupes on
``timestamp`` with first-occurrence-wins.

In practice: write `fetch()` as if it's called once per window. The
engine makes it idempotent by construction.

## Know your canonical schema before you write `fetch()`

For SENTIMENT, the canonical schema is:

```python
from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataType

CANONICAL_SCHEMAS[DataType.SENTIMENT]
# IndexValueSchema — requires:
#   timestamp: datetime64[us, UTC], unique, non-null, sorted ascending
#   value:     float, non-null
#   (strict=True: NO extra columns)
#   (no range constraint on `value` — source may impose stricter
#    checks inside fetch() if it wants, e.g. 0..1)
```

Other data types have different column names (OHLCV wants
``[timestamp, open, high, low, close, volume]`` with price > 0 and
volume >= 0; FUNDING wants ``[timestamp, funding_rate]``). Look up
your target schema in `crypto_alpha_engine.data.schemas` before
writing `fetch()`.

## A worked example: proprietary sentiment API

Internal REST API that scores tweets 0.0..1.0 per hour. Complete
source in about 50 lines including imports, retry handling, and
registration:

```python
# my_pkg/sources/mytwitter.py
import os
import time

import pandas as pd
import requests

from crypto_alpha_engine.data.protocol import DataType
from crypto_alpha_engine.data.registry import register_source


class MyTwitterSentiment:
    name = "mytwitter_v1"
    data_type = DataType.SENTIMENT
    symbols = ["BTC", "ETH"]

    def __init__(self, api_key: str, base_url: str = "https://sentiment.internal") -> None:
        self._key = api_key
        self._base = base_url

    def fetch(self, symbol, *, start, end=None, freq="1h"):
        params = {
            "symbol": symbol,
            "since": int(start.timestamp() * 1000),
            "until": None if end is None else int(end.timestamp() * 1000),
            "interval": freq,
        }
        # Minimal 429-retry: the failure contributors hit first.
        for attempt in range(3):
            resp = requests.get(
                f"{self._base}/scores",
                params=params,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=30,
            )
            if resp.status_code != 429:
                break
            time.sleep(2**attempt)
        resp.raise_for_status()
        rows = resp.json()["scores"]
        ts = pd.to_datetime([r["ts"] for r in rows], unit="ms", utc=True)
        return pd.DataFrame(
            {"timestamp": ts.astype("datetime64[us, UTC]"),
             "value": [float(r["score"]) for r in rows]}
        )

    def earliest_available(self, symbol):
        # Hardcode only as a last resort — query the API's metadata
        # endpoint if it exposes one (most do: /coverage, /about, etc.).
        return pd.Timestamp("2022-01-01", tz="UTC")


register_source(MyTwitterSentiment(api_key=os.environ["MYTWITTER_KEY"]))
```

## Using your source

Once registered, load through the generic source-aware loader:

```python
from crypto_alpha_engine.data.loader import load_by_source
from crypto_alpha_engine.data.protocol import DataType

df = load_by_source(DataType.SENTIMENT, "BTC", source="mytwitter_v1")
```

Or let the registry pick the default for ``(DataType.SENTIMENT, "BTC")``:

```python
df = load_by_source(DataType.SENTIMENT, "BTC")   # first-registered source
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
- [ ] Validated `fetch()` output locally against the canonical schema
      before registering. One-liner:
      `CANONICAL_SCHEMAS[DataType.SENTIMENT].validate(your_df)`
- [ ] Tests: at minimum, a happy-path test + one failure mode
      (HTTP error, malformed response).

That's it — no engine changes needed.

"""Extensibility interface for data sources.

Every dataset that reaches the engine is produced by a **data source**
that implements the :class:`DataSource` Protocol. The engine itself knows
nothing about specific exchanges or APIs — it asks the registry for a
source by *data type* (OHLCV, FUNDING, etc.) and *symbol*, and dispatches
the fetch. This keeps the engine open-source-friendly while letting
users add private or proprietary sources without touching engine code.

See SPEC §5.1 for the architectural description and
``docs/adding_custom_sources.md`` for a worked example of a custom
source in < 50 lines.

Invariants
----------

* **Source names are unique** across a single process. The registry
  enforces this at ``register_source`` time.
* **Schema ownership belongs to the engine.** One canonical Pandera
  schema per :class:`DataType`, exposed via :data:`CANONICAL_SCHEMAS`.
  A source's ``fetch()`` output MUST conform; the downloader validates
  before writing and quarantines on failure.
* **One source → one on-disk path.** Each source writes to
  ``data/<data_type>/<source_name>/...``; two sources for the same
  ``(data_type, symbol)`` write to different directories and never
  share a file. The loader checks file-level metadata on read to catch
  hand-moved files that violate this.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd
import pandera.pandas as pa

from crypto_alpha_engine.data.schemas import (
    FundingRateSchema,
    IndexValueSchema,
    OHLCVSchema,
    OpenInterestSchema,
)


class DataType(StrEnum):
    """Kinds of data the engine can consume.

    The value doubles as the directory segment under ``data/``, so
    ``DataType.OHLCV.value`` is the literal string ``"ohlcv"``.
    """

    OHLCV = "ohlcv"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    MACRO = "macro"


CANONICAL_SCHEMAS: dict[DataType, type[pa.DataFrameModel]] = {
    DataType.OHLCV: OHLCVSchema,
    DataType.FUNDING: FundingRateSchema,
    DataType.OPEN_INTEREST: OpenInterestSchema,
    DataType.ONCHAIN: IndexValueSchema,
    DataType.SENTIMENT: IndexValueSchema,
    DataType.MACRO: IndexValueSchema,
}
"""The canonical on-disk schema for each :class:`DataType`.

A source may impose stricter checks inside its own ``fetch()`` (e.g. Fear
& Greed's 0..100 bound), but the parquet that lands on disk must match
the canonical schema here. This is the contract the engine relies on.
"""


@runtime_checkable
class DataSource(Protocol):
    """The interface every data source (built-in or custom) implements.

    Attributes are intentionally simple (no methods to compute them)
    because sources are typically small classes with static declarations.

    Attributes:
        name: Unique identifier. Used as the subdirectory under
            ``data/<data_type>/`` and as the ``source_name`` key in
            parquet file metadata.
        data_type: Which canonical schema this source's output must
            conform to.
        symbols: Symbols this source supports. The registry filters on
            this list when resolving defaults.

    Methods:
        fetch: Return a DataFrame for ``symbol`` over ``[start, end)``
            at ``freq`` resolution. Must conform to the canonical
            schema for :attr:`data_type`. Pagination, retries, and
            auth are all the source's concern.
        earliest_available: Lower bound for ``symbol`` — the source
            promises ``fetch()`` will not return rows older than this.
            The orchestrator uses it to avoid asking for impossible
            windows.
    """

    name: str
    data_type: DataType
    symbols: list[str]

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame: ...

    def earliest_available(self, symbol: str) -> pd.Timestamp: ...

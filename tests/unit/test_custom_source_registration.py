"""End-to-end test that a custom DataSource flows through the engine.

This proves the extensibility claim in SPEC §5.1: a user who doesn't
touch engine code can add a source, run a download, and load it back —
all via the standard public APIs. If this test passes, the abstraction
works.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_alpha_engine.data import registry as registry_mod
from crypto_alpha_engine.data.downloader import run_source
from crypto_alpha_engine.data.loader import load_by_source
from crypto_alpha_engine.data.protocol import DataType

# ---------------------------------------------------------------------------
# Mock "proprietary sentiment" source — modeled on
# docs/adding_custom_sources.md but self-contained (no HTTP).
# ---------------------------------------------------------------------------


class MockProprietarySentiment:
    """Pretend internal API that scores tweets 0.0..1.0 per hour."""

    name = "mock_proprietary_sentiment_v1"
    data_type = DataType.SENTIMENT
    symbols = ["BTC", "ETH"]

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame:
        # Deterministic synthetic data covering [start, end or start+3h].
        if end is None:
            end = start + pd.Timedelta("3h")
        ts = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        return pd.DataFrame(
            {
                "timestamp": ts,
                "value": [0.42 + 0.01 * i for i in range(len(ts))],
            }
        )

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        return pd.Timestamp("2024-01-01", tz="UTC")


def test_custom_source_flows_end_to_end(tmp_path: Path) -> None:
    """Register → run_source → load_by_source: the standard public flow."""
    registry_mod._reset_for_tests()  # isolate from any built-ins

    src = MockProprietarySentiment()
    registry_mod.register_source(src)

    # 1. Download via the orchestrator.
    out_path = run_source(
        src,
        "BTC",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
        freq="1h",
        data_dir=tmp_path,
    )
    assert out_path is not None
    # Path is under data/<data_type>/<source_name>/ — the canonical layout.
    assert out_path.parent == (tmp_path / "sentiment" / "mock_proprietary_sentiment_v1")

    # 2. Load it back through the source-aware loader.
    df = load_by_source(
        DataType.SENTIMENT,
        "BTC",
        source="mock_proprietary_sentiment_v1",
        data_dir=tmp_path,
    )
    assert len(df) == 5
    assert df["value"].iloc[0] == 0.42

    # 3. Loading without an explicit `source=` uses the registry default —
    # since we only registered one source, it resolves to our mock.
    df_default = load_by_source(
        DataType.SENTIMENT,
        "BTC",
        data_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(df, df_default)

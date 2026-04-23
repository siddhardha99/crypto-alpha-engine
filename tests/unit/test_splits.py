"""Unit tests for the splits module.

Covers the explicit-flag behavior and reveal-logging contract from SPEC §5,
plus DataSplits immutability. The no-leakage properties live in
``tests/property/test_splits_no_leakage.py``.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from crypto_alpha_engine.data.splits import (
    DEFAULT_SPLITS,
    DataSplits,
    split_test,
    split_train,
    split_train_plus_validation,
    split_validation,
)
from crypto_alpha_engine.exceptions import ConfigError

# ---------------------------------------------------------------------------
# A small fixed frame spanning all three splits.
# ---------------------------------------------------------------------------


@pytest.fixture
def tri_split_frame() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            # Train zone
            "2020-06-15 00:00:00",
            "2023-12-31 23:00:00",
            # Validation zone
            "2024-01-01 00:00:00",
            "2024-06-30 12:00:00",
            "2024-12-31 23:00:00",
            # Test zone
            "2025-01-01 00:00:00",
            "2025-07-01 00:00:00",
        ],
        utc=True,
    ).astype("datetime64[us, UTC]")
    return pd.DataFrame({"timestamp": ts, "value": range(len(ts))})


# ---------------------------------------------------------------------------
# DataSplits dataclass
# ---------------------------------------------------------------------------


class TestDataSplits:
    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_SPLITS.train_end = pd.Timestamp("2030-01-01", tz="UTC")  # type: ignore[misc]

    def test_rejects_non_utc_train_end(self) -> None:
        with pytest.raises(ConfigError, match="train_end"):
            DataSplits(
                train_end=pd.Timestamp("2024-01-01"),  # naive
                validation_end=pd.Timestamp("2025-01-01", tz="UTC"),
            )

    def test_rejects_non_utc_validation_end(self) -> None:
        with pytest.raises(ConfigError, match="validation_end"):
            DataSplits(
                train_end=pd.Timestamp("2024-01-01", tz="UTC"),
                validation_end=pd.Timestamp("2025-01-01"),  # naive
            )

    def test_rejects_validation_end_before_train_end(self) -> None:
        with pytest.raises(ConfigError, match="validation_end"):
            DataSplits(
                train_end=pd.Timestamp("2025-01-01", tz="UTC"),
                validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
            )


# ---------------------------------------------------------------------------
# Happy-path splits
# ---------------------------------------------------------------------------


class TestSplits:
    def test_split_train_contains_only_pre_train_end(self, tri_split_frame: pd.DataFrame) -> None:
        train = split_train(tri_split_frame)
        assert len(train) == 2
        assert train["timestamp"].max() < DEFAULT_SPLITS.train_end

    def test_split_validation_is_half_open(self, tri_split_frame: pd.DataFrame) -> None:
        """Validation includes train_end, excludes validation_end."""
        val = split_validation(tri_split_frame)
        assert len(val) == 3
        assert val["timestamp"].min() >= DEFAULT_SPLITS.train_end
        assert val["timestamp"].max() < DEFAULT_SPLITS.validation_end

    def test_split_train_plus_validation_is_concatenation(
        self, tri_split_frame: pd.DataFrame
    ) -> None:
        tpv = split_train_plus_validation(tri_split_frame)
        assert len(tpv) == 5  # 2 train + 3 validation
        assert tpv["timestamp"].max() < DEFAULT_SPLITS.validation_end

    def test_accepts_custom_splits(self, tri_split_frame: pd.DataFrame) -> None:
        custom = DataSplits(
            train_end=pd.Timestamp("2022-01-01", tz="UTC"),
            validation_end=pd.Timestamp("2023-01-01", tz="UTC"),
        )
        train = split_train(tri_split_frame, splits=custom)
        assert len(train) == 1  # only 2020-06-15


# ---------------------------------------------------------------------------
# split_test — the guardrail
# ---------------------------------------------------------------------------


class TestSplitTest:
    def test_requires_reveal_flag(self, tri_split_frame: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="reveal_test_set"):
            split_test(tri_split_frame, reveal_test_set=False, reason="n/a")

    def test_requires_non_empty_reason(self, tri_split_frame: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="reason"):
            split_test(tri_split_frame, reveal_test_set=True, reason="")

    def test_requires_reason_not_whitespace_only(self, tri_split_frame: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="reason"):
            split_test(tri_split_frame, reveal_test_set=True, reason="   ")

    def test_returns_test_zone_rows_when_authorized(self, tri_split_frame: pd.DataFrame) -> None:
        test = split_test(tri_split_frame, reveal_test_set=True, reason="final evaluation run")
        assert len(test) == 2
        assert test["timestamp"].min() >= DEFAULT_SPLITS.validation_end

    def test_access_is_logged(
        self,
        tri_split_frame: pd.DataFrame,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SPEC §5: every test-set reveal must be logged, with the reason."""
        from crypto_alpha_engine.data import splits as splits_mod

        captured_events: list[dict[str, object]] = []
        original = splits_mod._log_test_reveal

        def _capturing(reason: str, n_rows: int) -> None:
            captured_events.append({"reason": reason, "n_rows": n_rows})
            original(reason, n_rows)

        splits_mod._log_test_reveal = _capturing
        try:
            split_test(
                tri_split_frame,
                reveal_test_set=True,
                reason="final evaluation run",
            )
        finally:
            splits_mod._log_test_reveal = original

        _ = capsys  # silence unused-fixture warning; kept for future log capture
        assert len(captured_events) == 1
        assert captured_events[0]["reason"] == "final evaluation run"
        assert captured_events[0]["n_rows"] == 2

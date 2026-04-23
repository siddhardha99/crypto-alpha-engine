"""Property tests for data splits — no timestamp ever crosses a boundary.

These are the tests SPEC §13 calls out by name for the splits module: no
matter how the input data is shaped (as long as it's valid), the train /
validation / test slices must be disjoint and ordered in time. Each
property is expressed as an invariant over hypothesis-generated input
so small edge cases (single row, all rows in one split, duplicates near
the boundary) get exercised automatically.

Written BEFORE ``splits.py`` exists, per CLAUDE.md Rule 2 (TDD) and the
user's explicit instruction to get the no-leakage properties failing
first.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crypto_alpha_engine.data.splits import (
    DEFAULT_SPLITS,
    split_test,
    split_train,
    split_train_plus_validation,
    split_validation,
)

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_SPLIT_BOUNDARY_MIN = datetime(2017, 1, 1, tzinfo=UTC)
_SPLIT_BOUNDARY_MAX = datetime(2027, 1, 1, tzinfo=UTC)


def _utc_datetime_strategy() -> st.SearchStrategy[datetime]:
    """UTC-aware datetimes across a window that spans all three splits."""
    return st.datetimes(
        min_value=_SPLIT_BOUNDARY_MIN.replace(tzinfo=None),
        max_value=_SPLIT_BOUNDARY_MAX.replace(tzinfo=None),
    ).map(lambda dt: dt.replace(tzinfo=UTC))


@st.composite
def _timestamp_frame(draw: st.DrawFn, min_size: int = 2, max_size: int = 200) -> pd.DataFrame:
    ts_list = draw(
        st.lists(
            _utc_datetime_strategy(),
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    ts_list.sort()
    # Canonical on-disk / in-memory form: datetime64[us, UTC].
    ts = pd.to_datetime(ts_list, utc=True).astype("datetime64[us, UTC]")
    return pd.DataFrame({"timestamp": ts, "value": range(len(ts_list))})


# ---------------------------------------------------------------------------
# Core no-leakage invariants (SPEC §13)
# ---------------------------------------------------------------------------


@given(df=_timestamp_frame())
@settings(max_examples=200)
def test_train_and_validation_are_disjoint(df: pd.DataFrame) -> None:
    """Property from SPEC §13: no timestamp appears in both train and validation."""
    train = split_train(df)
    validation = split_validation(df)
    assert set(train["timestamp"]).isdisjoint(set(validation["timestamp"]))


@given(df=_timestamp_frame())
@settings(max_examples=200)
def test_train_and_test_are_disjoint(df: pd.DataFrame) -> None:
    """Property from SPEC §13: no timestamp in the test set also appears in train."""
    train = split_train(df)
    test = split_test(df, reveal_test_set=True, reason="property test")
    assert set(train["timestamp"]).isdisjoint(set(test["timestamp"]))


@given(df=_timestamp_frame())
@settings(max_examples=200)
def test_validation_and_test_are_disjoint(df: pd.DataFrame) -> None:
    """Closes the loop: val and test also never share a timestamp."""
    validation = split_validation(df)
    test = split_test(df, reveal_test_set=True, reason="property test")
    assert set(validation["timestamp"]).isdisjoint(set(test["timestamp"]))


@given(df=_timestamp_frame())
@settings(max_examples=200)
def test_max_train_strictly_before_min_validation(df: pd.DataFrame) -> None:
    """Property from SPEC §13: max(train.timestamp) < min(validation.timestamp)."""
    train = split_train(df)
    validation = split_validation(df)
    if not train.empty and not validation.empty:
        assert train["timestamp"].max() < validation["timestamp"].min()


@given(df=_timestamp_frame())
@settings(max_examples=200)
def test_max_validation_strictly_before_min_test(df: pd.DataFrame) -> None:
    """Extends the ordering property to the validation → test boundary."""
    validation = split_validation(df)
    test = split_test(df, reveal_test_set=True, reason="property test")
    if not validation.empty and not test.empty:
        assert validation["timestamp"].max() < test["timestamp"].min()


# ---------------------------------------------------------------------------
# Coverage: every row lands in exactly one split.
# ---------------------------------------------------------------------------


@given(df=_timestamp_frame())
@settings(max_examples=200)
def test_splits_cover_all_input_rows(df: pd.DataFrame) -> None:
    """Every row in the input appears in exactly one of the three splits."""
    train = split_train(df)
    validation = split_validation(df)
    test = split_test(df, reveal_test_set=True, reason="property test")
    total = len(train) + len(validation) + len(test)
    assert total == len(df)


@given(df=_timestamp_frame())
@settings(max_examples=100)
def test_train_plus_validation_matches_concat(df: pd.DataFrame) -> None:
    """split_train_plus_validation is exactly the row-wise concat of the two."""
    combined = split_train_plus_validation(df)
    train = split_train(df)
    validation = split_validation(df)
    assert len(combined) == len(train) + len(validation)
    assert set(combined["timestamp"]) == set(train["timestamp"]) | set(validation["timestamp"])


# ---------------------------------------------------------------------------
# DEFAULT_SPLITS sanity: the shipped boundaries match SPEC §5.
# ---------------------------------------------------------------------------


def test_default_splits_boundaries_are_utc_and_ordered() -> None:
    assert DEFAULT_SPLITS.train_end.tzinfo is not None
    assert DEFAULT_SPLITS.validation_end.tzinfo is not None
    assert DEFAULT_SPLITS.train_end < DEFAULT_SPLITS.validation_end


def test_default_splits_match_spec_section_5() -> None:
    """SPEC §5: train through 2023-12-31, validation Jan-Dec 2024, test from 2025-01-01."""
    assert DEFAULT_SPLITS.train_end == pd.Timestamp("2024-01-01", tz="UTC")
    assert DEFAULT_SPLITS.validation_end == pd.Timestamp("2025-01-01", tz="UTC")

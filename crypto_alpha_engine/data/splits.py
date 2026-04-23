"""Train / validation / test split boundaries and enforcement.

SPEC §5 partitions every dataset into three half-open time windows:

* **Train:**       ``[-inf, train_end)``
* **Validation:**  ``[train_end, validation_end)``
* **Test:**        ``[validation_end, +inf)``

All boundaries are UTC timestamps. The module exposes:

* :class:`DataSplits` — a frozen dataclass holding the two boundaries.
* :data:`DEFAULT_SPLITS` — the shipped boundaries from SPEC §5
  (train ends 2024-01-01 UTC, validation ends 2025-01-01 UTC).
* :func:`split_train`, :func:`split_validation`,
  :func:`split_train_plus_validation` — pure slicing helpers.
* :func:`split_test` — the gated test-set accessor. Requires an explicit
  ``reveal_test_set=True`` and a non-empty ``reason``, and logs every
  successful reveal via structlog (SPEC §5: "reveal requires an explicit
  flag that is logged").

Timestamps in input DataFrames are expected to live in a ``timestamp``
column with dtype ``datetime64[us, UTC]`` (the canonical form per
``crypto_alpha_engine.data.schemas``). The slicing functions don't validate
the schema themselves — that's the loader's job — but they do require a
``timestamp`` column and will raise ``KeyError`` if it's missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from crypto_alpha_engine.exceptions import ConfigError

if TYPE_CHECKING:
    pass

_logger = structlog.get_logger(__name__)

_TIMESTAMP_COL = "timestamp"


# ---------------------------------------------------------------------------
# DataSplits
# ---------------------------------------------------------------------------


def _default_train_end() -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="UTC")


def _default_validation_end() -> pd.Timestamp:
    return pd.Timestamp("2025-01-01", tz="UTC")


@dataclass(frozen=True)
class DataSplits:
    """The two boundary timestamps that define the train / val / test partition.

    Boundaries are half-open: a row with ``timestamp == train_end`` belongs
    to the validation split, not train. This avoids off-by-one ambiguity at
    the boundary.

    Args:
        train_end: UTC timestamp. Rows with ``timestamp < train_end`` are
            train; rows with ``timestamp >= train_end`` are not.
        validation_end: UTC timestamp. Rows with
            ``train_end <= timestamp < validation_end`` are validation; rows
            with ``timestamp >= validation_end`` are test.

    Raises:
        ConfigError: If either boundary is naive (no tzinfo), or if
            ``validation_end <= train_end``.

    Example:
        >>> DEFAULT_SPLITS.train_end
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    """

    train_end: pd.Timestamp = field(default_factory=_default_train_end)
    validation_end: pd.Timestamp = field(default_factory=_default_validation_end)

    def __post_init__(self) -> None:
        if self.train_end.tzinfo is None:
            raise ConfigError(
                f"DataSplits.train_end must be UTC-aware; got naive {self.train_end!r}"
            )
        if self.validation_end.tzinfo is None:
            raise ConfigError(
                "DataSplits.validation_end must be UTC-aware; got naive "
                f"{self.validation_end!r}"
            )
        if self.validation_end <= self.train_end:
            raise ConfigError(
                f"DataSplits.validation_end ({self.validation_end!r}) must be strictly "
                f"after train_end ({self.train_end!r})"
            )


DEFAULT_SPLITS = DataSplits()
"""SPEC §5 shipped boundaries: train through 2023, validation 2024, test 2025+."""


# ---------------------------------------------------------------------------
# Public slicing functions
# ---------------------------------------------------------------------------


def split_train(df: pd.DataFrame, *, splits: DataSplits = DEFAULT_SPLITS) -> pd.DataFrame:
    """Return the rows strictly before ``splits.train_end``.

    Args:
        df: DataFrame with a UTC-aware ``timestamp`` column.
        splits: Boundary configuration. Defaults to ``DEFAULT_SPLITS``.

    Returns:
        A new DataFrame containing only train-zone rows. Order is preserved;
        the index is reset so consumers can rely on ``0..n-1``.
    """
    return _slice_range(df, upper_exclusive=splits.train_end)


def split_validation(
    df: pd.DataFrame, *, splits: DataSplits = DEFAULT_SPLITS
) -> pd.DataFrame:
    """Return the validation-zone rows: ``[train_end, validation_end)``."""
    return _slice_range(
        df,
        lower_inclusive=splits.train_end,
        upper_exclusive=splits.validation_end,
    )


def split_train_plus_validation(
    df: pd.DataFrame, *, splits: DataSplits = DEFAULT_SPLITS
) -> pd.DataFrame:
    """Return train + validation as a single DataFrame (for final-model fit).

    Equivalent to concatenating :func:`split_train` and :func:`split_validation`;
    implemented as a single slice for efficiency.
    """
    return _slice_range(df, upper_exclusive=splits.validation_end)


def split_test(
    df: pd.DataFrame,
    *,
    reveal_test_set: bool,
    reason: str,
    splits: DataSplits = DEFAULT_SPLITS,
) -> pd.DataFrame:
    """Return the test-zone rows, gated by an explicit reveal flag.

    Per SPEC §5 and Principle 3 (data splits are sacred), the test set is
    off-limits during research. Callers must pass ``reveal_test_set=True``
    and a non-empty ``reason`` describing why they need it (e.g. "final
    evaluation run for paper"). Every successful reveal is logged.

    Args:
        df: DataFrame with a UTC-aware ``timestamp`` column.
        reveal_test_set: Must be ``True``. Passing ``False`` raises.
        reason: Non-whitespace explanation of why the test set is being
            accessed. Becomes part of the log record.
        splits: Boundary configuration. Defaults to ``DEFAULT_SPLITS``.

    Returns:
        A DataFrame of test-zone rows.

    Raises:
        ConfigError: If ``reveal_test_set`` is False, or if ``reason`` is
            empty or whitespace-only.

    Example:
        >>> # test_df = split_test(df, reveal_test_set=True, reason="final eval")
    """
    if not reveal_test_set:
        raise ConfigError(
            "split_test requires reveal_test_set=True (SPEC §5 / Principle 3); "
            "research code should use split_train or split_validation instead"
        )
    if not reason or not reason.strip():
        raise ConfigError(
            "split_test requires a non-empty `reason` so every test-set "
            "reveal has an explanation in the audit log"
        )

    test = _slice_range(df, lower_inclusive=splits.validation_end)
    _log_test_reveal(reason=reason.strip(), n_rows=len(test))
    return test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slice_range(
    df: pd.DataFrame,
    *,
    lower_inclusive: pd.Timestamp | None = None,
    upper_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Half-open slice by the ``timestamp`` column.

    Either bound may be omitted to leave that side unbounded.
    """
    if _TIMESTAMP_COL not in df.columns:
        raise KeyError(
            f"expected a '{_TIMESTAMP_COL}' column; got {list(df.columns)!r}"
        )
    ts = df[_TIMESTAMP_COL]
    mask: pd.Series = pd.Series(True, index=df.index)
    if lower_inclusive is not None:
        mask &= ts >= lower_inclusive
    if upper_exclusive is not None:
        mask &= ts < upper_exclusive
    return df.loc[mask].reset_index(drop=True)


def _log_test_reveal(reason: str, n_rows: int) -> None:
    """Emit a structured log event for a successful test-set reveal.

    Defined as a module-level function so tests can monkey-patch it to
    assert the reveal happened without touching the underlying structlog
    configuration.
    """
    _logger.warning(
        "test_set_revealed",
        reason=reason,
        n_rows=n_rows,
    )

"""Append-only JSONL experiment ledger (SPEC §11).

One line per completed backtest. Each line is a self-contained JSON
object with three buckets:

* ``schema_version`` — integer version tag (per-line, not per-file,
  so mixed-version files are supported during a future migration).
* ``meta`` — write-time provenance: UTC timestamp, engine version,
  UUID for cross-referencing.
* ``factor`` — the :class:`Factor` that produced the result, with
  its AST serialized via :func:`canonical_form`. Enough to replay
  the experiment or compare structurally against future candidates.
* ``result`` — flat :class:`BacktestResult` fields. Numeric fields
  that can legitimately carry ``inf``/``-inf``/``nan`` (per the
  Phase-5 metrics contract) are encoded as sentinel strings
  ``"inf"`` / ``"-inf"`` / ``"nan"``. Sentinel substitution is
  restricted to known numeric fields so a factor_name literally
  equal to ``"inf"`` isn't confused.

Atomicity: appends open the file in text-append mode and write one
line + newline + flush. Good enough for single-user research
workloads (SPEC §11 scope). Multi-writer concurrency is out of scope
for v1.0.

Query surface
-------------

* :meth:`Ledger.append` — add one entry.
* :meth:`Ledger.read_all` — iterator over every entry.
* :meth:`Ledger.count_experiments` — literal count (includes
  NaN-Sharpe entries; failed backtests are still trials).
* :meth:`Ledger.count_finite_experiments` — count filtered to
  ``math.isfinite(sharpe)``. Safe input to DSR's ``n_trials``.
* :meth:`Ledger.sharpe_variance_across_trials` — sample variance of
  Sharpe across finite-Sharpe entries. Returns ``NaN`` on fewer
  than 2 finite entries.

See ``docs/methodology.md`` for the rationale on why count and
variance have different NaN filtering — NaN Sharpes are meaningful
trials for multi-testing correction but can't contribute to variance
computation.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.factor.ast import canonical_form
from crypto_alpha_engine.types import BacktestResult, Factor, FactorNode

SCHEMA_VERSION: int = 1
"""The ledger's current JSONL schema version. Written into every line
for forward-compatibility during future migrations. Read-time parser
raises on any unknown version."""

try:
    _ENGINE_VERSION: str = importlib.metadata.version("crypto-alpha-engine")
except importlib.metadata.PackageNotFoundError:
    _ENGINE_VERSION = "unknown"


# ---------------------------------------------------------------------------
# BacktestResult field-type taxonomy (drives sentinel encoding)
# ---------------------------------------------------------------------------

_STRING_RESULT_FIELDS: frozenset[str] = frozenset({"factor_id", "factor_name", "data_version"})
"""Result fields whose values are plain strings. Passed through JSON
as-is; NOT subject to sentinel substitution even if a user's
factor_name happens to equal ``"inf"``."""

_DATETIME_RESULT_FIELDS: frozenset[str] = frozenset({"run_timestamp"})
"""Result fields serialized as ISO-8601 strings."""

_INT_RESULT_FIELDS: frozenset[str] = frozenset(
    {
        "n_experiments_in_ledger",
        "factor_ast_depth",
        "factor_node_count",
        "n_trades",
    }
)
"""Integer-valued result fields. No sentinel encoding needed (ints
are always JSON-representable)."""


def _is_numeric_field(field_name: str) -> bool:
    """Result field eligible for inf/nan sentinel encoding."""
    return field_name not in (_STRING_RESULT_FIELDS | _DATETIME_RESULT_FIELDS | _INT_RESULT_FIELDS)


# ---------------------------------------------------------------------------
# Ledger entry (parsed line)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    """One parsed ledger line.

    Attributes:
        schema_version: The ledger line's declared schema version.
        meta: Dict of write-time provenance keys — ``written_at``,
            ``engine_version``, ``ledger_line_id``. Kept as a dict
            (not a dataclass) so future schema revisions can add
            keys without changing this class.
        factor: The reconstructed :class:`Factor`. The AST is
            reconstructed from its canonical-form dict.
        result: The reconstructed :class:`BacktestResult`. Sentinel
            strings are decoded back to inf/nan floats.
    """

    schema_version: int
    meta: dict[str, str]
    factor: Factor
    result: BacktestResult


# ---------------------------------------------------------------------------
# Ledger (the public surface)
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only JSONL ledger at ``path``.

    The file is created lazily on first :meth:`append`. Reading a
    nonexistent file yields nothing — an empty ledger is a valid
    initial state.

    Example:
        >>> ledger = Ledger(Path("ledger.jsonl"))
        >>> ledger.append(factor=my_factor, result=my_result)
        >>> ledger.count_experiments()
        1
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, *, factor: Factor, result: BacktestResult) -> None:
        """Serialize one (factor, result) pair and append to the ledger.

        The entry's ``ledger_line_id`` is a fresh UUID4, so cross-run
        references remain unique even if factor_id or factor_name
        collide.

        Args:
            factor: The factor that produced the result.
            result: The :class:`BacktestResult` to record.

        Raises:
            ValueError: If any numeric field contains a non-finite
                float that slipped past the sentinel encoder (defense
                in depth — should never happen in practice).
        """
        entry = {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "written_at": datetime.now(UTC).isoformat(),
                "engine_version": _ENGINE_VERSION,
                "ledger_line_id": str(uuid.uuid4()),
            },
            "factor": _encode_factor(factor),
            "result": _encode_result_dict(result),
        }
        line = json.dumps(
            entry,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(line + "\n")

    def read_all(self) -> Iterator[LedgerEntry]:
        """Yield every entry in insertion order.

        Raises:
            ConfigError: On a line whose ``schema_version`` is unknown
                (current parser only understands SCHEMA_VERSION=1).
        """
        if not self.path.exists():
            return
        with self.path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield _parse_entry(data)

    def count_experiments(
        self,
        filter_fn: Callable[[LedgerEntry], bool] | None = None,
    ) -> int:
        """Literal count of ledger entries (no Sharpe filter).

        Includes entries with NaN or infinite Sharpe — failed
        backtests are still trials in the multiple-testing sense, so
        they must count toward ``n_trials`` for DSR correction.

        Use :meth:`count_finite_experiments` if you need a count that
        matches the one used in variance computation.

        Args:
            filter_fn: Optional predicate on :class:`LedgerEntry`.
                Default ``None`` counts every entry.

        Returns:
            Integer count.
        """
        return sum(1 for e in self.read_all() if filter_fn is None or filter_fn(e))

    def count_finite_experiments(
        self,
        filter_fn: Callable[[LedgerEntry], bool] | None = None,
    ) -> int:
        """Count of entries with finite Sharpe.

        Applies the same filter used by
        :meth:`sharpe_variance_across_trials`. Users who understand
        DSR semantics feed this to ``n_trials``; users who don't
        default to :meth:`count_experiments` and accept the
        slightly-conservative multiple-testing penalty.

        Do NOT "fix" the asymmetry by also filtering
        :meth:`count_experiments` — see the module docstring. NaN
        Sharpes are meaningful as trials but can't contribute to
        variance.
        """
        return sum(
            1
            for e in self.read_all()
            if (filter_fn is None or filter_fn(e)) and math.isfinite(e.result.sharpe)
        )

    def sharpe_variance_across_trials(
        self,
        filter_fn: Callable[[LedgerEntry], bool] | None = None,
    ) -> float:
        """Sample variance (ddof=1) of Sharpe across finite-Sharpe entries.

        Returns ``NaN`` when fewer than 2 finite-Sharpe entries pass
        the filter. That NaN flows through
        :func:`deflated_sharpe_ratio` as NaN DSR, which is the
        correct "ledger has no useful variance info" signal.

        Non-finite Sharpes (NaN from zero-vol runs, inf from edge
        cases) are excluded from the variance computation because
        ``np.var`` propagates NaN through to the result. Documented
        here as the intentional asymmetry — see
        :meth:`count_experiments` for the counterpart that does NOT
        filter.
        """
        sharpes = [
            e.result.sharpe
            for e in self.read_all()
            if (filter_fn is None or filter_fn(e)) and math.isfinite(e.result.sharpe)
        ]
        if len(sharpes) < 2:
            return float("nan")
        return float(np.var(sharpes, ddof=1))


# ---------------------------------------------------------------------------
# Serialization — encode
# ---------------------------------------------------------------------------


def _encode_result_dict(result: BacktestResult) -> dict[str, Any]:
    """Convert a :class:`BacktestResult` to a JSON-safe dict.

    Numeric fields with ``inf``/``-inf``/``nan`` values are encoded
    as sentinel strings. String, integer, and datetime fields pass
    through unchanged (datetime → ISO string).
    """
    out: dict[str, Any] = {}
    for field in fields(BacktestResult):
        value = getattr(result, field.name)
        if field.name in _DATETIME_RESULT_FIELDS:
            out[field.name] = cast(datetime, value).isoformat()
        elif field.name in _STRING_RESULT_FIELDS or field.name in _INT_RESULT_FIELDS:
            out[field.name] = value
        else:
            out[field.name] = _encode_numeric(cast(float, value))
    return out


def _encode_numeric(v: float) -> float | str:
    """Return a JSON-safe representation of a float.

    Sentinel strings: ``"inf"`` for ``+inf``, ``"-inf"`` for ``-inf``,
    ``"nan"`` for ``nan``. Finite floats pass through unchanged.
    """
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _encode_factor(factor: Factor) -> dict[str, Any]:
    """Serialize a :class:`Factor` to a JSON-safe dict."""
    return {
        "name": factor.name,
        "description": factor.description,
        "hypothesis": factor.hypothesis,
        "root": canonical_form(factor.root),
        "metadata": dict(factor.metadata),
    }


# ---------------------------------------------------------------------------
# Serialization — decode
# ---------------------------------------------------------------------------


def _decode_result_dict(data: dict[str, Any]) -> BacktestResult:
    """Reconstruct a :class:`BacktestResult` from an encoded dict."""
    kwargs: dict[str, Any] = {}
    for field in fields(BacktestResult):
        raw = data[field.name]
        if field.name in _DATETIME_RESULT_FIELDS:
            kwargs[field.name] = datetime.fromisoformat(raw)
        elif field.name in _STRING_RESULT_FIELDS:
            kwargs[field.name] = raw
        elif field.name in _INT_RESULT_FIELDS:
            kwargs[field.name] = int(raw)
        else:
            kwargs[field.name] = _decode_numeric(raw)
    return BacktestResult(**kwargs)


def _decode_numeric(raw: Any) -> float:
    """Inverse of :func:`_encode_numeric`."""
    if isinstance(raw, str):
        if raw == "nan":
            return float("nan")
        if raw == "inf":
            return float("inf")
        if raw == "-inf":
            return float("-inf")
        raise ConfigError(
            f"unknown numeric sentinel {raw!r}; expected one of "
            f"'nan' / 'inf' / '-inf' or a real number"
        )
    return float(raw)


def _decode_factor(data: dict[str, Any]) -> Factor:
    return Factor(
        name=data["name"],
        description=data["description"],
        hypothesis=data["hypothesis"],
        root=_decode_factor_node(data["root"]),
        metadata=dict(data.get("metadata", {})),
    )


def _decode_factor_node(data: dict[str, Any]) -> FactorNode:
    """Inverse of :func:`canonical_form`."""
    args = tuple(_decode_factor_node(a) if isinstance(a, dict) else a for a in data["args"])
    return FactorNode(operator=data["op"], args=args, kwargs=dict(data["kwargs"]))


def _parse_entry(data: dict[str, Any]) -> LedgerEntry:
    """Parse one JSON line into a :class:`LedgerEntry`."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported ledger schema_version {version!r}; "
            f"this build understands only version {SCHEMA_VERSION}. "
            f"Upgrade the engine or migrate the ledger file."
        )
    return LedgerEntry(
        schema_version=int(version),
        meta=dict(data["meta"]),
        factor=_decode_factor(data["factor"]),
        result=_decode_result_dict(data["result"]),
    )

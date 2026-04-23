"""Round-trip + aggregation tests for the experiment ledger.

Scope (commit 1 of Phase 7): pure I/O. No duplicate detection here
(commit 2), no engine integration (commit 3). This file pins:

* Serialization → deserialization is byte-round-trip for every
  :class:`BacktestResult` field, including the ``inf``/``nan``
  sentinel encoding.
* Count/variance semantics — ``count_experiments`` literal,
  ``count_finite_experiments`` filtered, variance filtered with
  the documented asymmetry.
* Schema version tagging + mixed-version file reading.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from crypto_alpha_engine.ledger.ledger import (
    SCHEMA_VERSION,
    Ledger,
    LedgerEntry,
    _decode_result_dict,
    _encode_result_dict,
)
from crypto_alpha_engine.types import BacktestResult, Factor, FactorNode


def _make_factor(name: str = "test_factor") -> Factor:
    root = FactorNode(operator="ts_mean", args=("BTC/USD|close", 20), kwargs={})
    return Factor(
        name=name,
        description=f"desc for {name}",
        hypothesis=f"hypothesis for {name}",
        root=root,
        metadata={"author": "test"},
    )


def _make_result(
    *,
    factor_id: str = "f_abc123",
    sharpe: float = 1.23,
    profit_factor_field: float = 1.5,
    complexity: float = 0.24,
    **overrides: object,
) -> BacktestResult:
    """Build a BacktestResult with every field populated."""
    kwargs: dict[str, object] = {
        "factor_id": factor_id,
        "factor_name": "test_factor",
        "run_timestamp": datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC),
        "data_version": "sha256:deadbeef",
        "sharpe": sharpe,
        "sortino": 1.45,
        "calmar": 0.87,
        "max_drawdown": -0.18,
        "annualized_return": 0.22,
        "total_return": 0.67,
        "ic_mean": 0.03,
        "ic_std": 0.12,
        "ic_ir": 0.25,
        "hit_rate": 0.54,
        "gross_sharpe": 1.55,
        "net_sharpe": 1.23,
        "turnover_annual": 8.4,
        "total_fees_paid": 1234.5,
        "total_slippage_paid": 678.9,
        "bull_sharpe": 1.8,
        "bear_sharpe": 0.4,
        "crab_sharpe": 0.9,
        "high_vol_sharpe": 1.1,
        "low_vol_sharpe": 1.4,
        "normal_vol_sharpe": 1.3,
        "euphoric_funding_sharpe": 0.7,
        "fearful_funding_sharpe": 1.6,
        "neutral_funding_sharpe": 1.2,
        "walk_forward_sharpe_mean": 1.15,
        "walk_forward_sharpe_std": 0.30,
        "in_sample_sharpe": 1.40,
        "out_of_sample_sharpe": 1.05,
        "deflated_sharpe_ratio": 0.78,
        "n_experiments_in_ledger": 42,
        "complexity_scalar": complexity,
        "factor_ast_depth": 4,
        "factor_node_count": 11,
        "factor_max_similarity_to_zoo": 0.33,
        "n_trades": 523,
        "avg_trade_duration_hours": 14.2,
        "avg_position_size": 0.25,
        "max_leverage_used": 1.5,
    }
    kwargs.update(overrides)
    # One named override for profit-factor-like inf testing:
    if "total_fees_paid" not in overrides:
        kwargs["total_fees_paid"] = profit_factor_field
    return BacktestResult(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sentinel encoding: inf / -inf / nan round-trip
# ---------------------------------------------------------------------------


class TestSentinelEncoding:
    def test_inf_encoded_as_string(self) -> None:
        d = _encode_result_dict(_make_result(total_fees_paid=float("inf")))
        assert d["total_fees_paid"] == "inf"

    def test_negative_inf_encoded_as_string(self) -> None:
        d = _encode_result_dict(_make_result(sharpe=float("-inf")))
        assert d["sharpe"] == "-inf"

    def test_nan_encoded_as_string(self) -> None:
        d = _encode_result_dict(_make_result(sharpe=float("nan")))
        assert d["sharpe"] == "nan"

    def test_string_fields_passed_through_even_if_they_look_like_sentinels(
        self,
    ) -> None:
        """factor_id is a string; a pathological value like "inf" must not
        be confused with the numeric sentinel on the way out OR on the
        way back in. Sentinel substitution is restricted to known
        numeric fields."""
        d = _encode_result_dict(_make_result(factor_id="f_inf_00000000", factor_name="inf"))
        assert d["factor_id"] == "f_inf_00000000"
        assert d["factor_name"] == "inf"

    def test_decode_inverts_encode(self) -> None:
        result = _make_result(
            sharpe=float("nan"),
            total_fees_paid=float("inf"),
            sortino=float("-inf"),
        )
        encoded = _encode_result_dict(result)
        decoded = _decode_result_dict(encoded)
        assert math.isnan(decoded.sharpe)
        assert math.isinf(decoded.total_fees_paid)
        assert decoded.total_fees_paid > 0
        assert math.isinf(decoded.sortino)
        assert decoded.sortino < 0

    def test_finite_floats_round_trip_exactly(self) -> None:
        result = _make_result(sharpe=1.23456789, sortino=-0.987654321)
        encoded = _encode_result_dict(result)
        decoded = _decode_result_dict(encoded)
        assert decoded.sharpe == 1.23456789
        assert decoded.sortino == -0.987654321


# ---------------------------------------------------------------------------
# Append + read-all round-trip
# ---------------------------------------------------------------------------


class TestAppendReadRoundTrip:
    def test_empty_ledger_reads_nothing(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "does_not_exist.jsonl")
        assert list(ledger.read_all()) == []

    def test_single_append_readback(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        factor = _make_factor()
        result = _make_result()
        ledger.append(factor=factor, result=result)

        entries = list(ledger.read_all())
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, LedgerEntry)
        assert entry.factor.name == factor.name
        assert entry.factor.description == factor.description
        assert entry.factor.hypothesis == factor.hypothesis
        assert entry.factor.root.operator == factor.root.operator
        assert entry.factor.root.args == factor.root.args
        assert entry.result.factor_id == result.factor_id
        assert entry.result.sharpe == result.sharpe
        assert entry.schema_version == SCHEMA_VERSION

    def test_multiple_appends_preserve_order(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        for i in range(5):
            ledger.append(
                factor=_make_factor(f"factor_{i}"),
                result=_make_result(factor_id=f"f_000{i}"),
            )
        entries = list(ledger.read_all())
        assert [e.result.factor_id for e in entries] == [f"f_000{i}" for i in range(5)]

    def test_every_backtest_result_field_round_trips(self, tmp_path: Path) -> None:
        """Contract check: no field gets silently dropped during
        serialize → deserialize."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        original = _make_result()
        ledger.append(factor=_make_factor(), result=original)

        (entry,) = list(ledger.read_all())
        roundtripped = entry.result
        for field_name in BacktestResult.__dataclass_fields__:
            original_value = getattr(original, field_name)
            rt_value = getattr(roundtripped, field_name)
            if isinstance(original_value, float) and math.isnan(original_value):
                assert math.isnan(rt_value)
            else:
                assert (
                    rt_value == original_value
                ), f"{field_name}: {rt_value!r} != {original_value!r}"

    def test_nested_factor_ast_round_trips(self, tmp_path: Path) -> None:
        """Multi-node factor: compiler-style nested AST survives
        serialize → deserialize intact, so duplicate detection in
        commit 2 has a working AST to compare against."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        inner = FactorNode(
            operator="ts_pct_change",
            args=("BTC/USD|close", 24),
            kwargs={},
        )
        root = FactorNode(operator="ts_zscore", args=(inner, 72), kwargs={})
        factor = Factor(
            name="nested",
            description="nested",
            hypothesis="h",
            root=root,
        )
        ledger.append(factor=factor, result=_make_result())
        (entry,) = list(ledger.read_all())

        # Round-tripped AST is structurally identical.
        assert entry.factor.root.operator == "ts_zscore"
        assert isinstance(entry.factor.root.args[0], FactorNode)
        assert entry.factor.root.args[0].operator == "ts_pct_change"
        assert entry.factor.root.args[0].args == ("BTC/USD|close", 24)

    def test_meta_block_carries_provenance(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        ledger.append(factor=_make_factor(), result=_make_result())
        (entry,) = list(ledger.read_all())
        assert "written_at" in entry.meta
        assert "engine_version" in entry.meta
        assert "ledger_line_id" in entry.meta
        # written_at parses as a UTC-aware datetime.
        dt = datetime.fromisoformat(entry.meta["written_at"])
        assert dt.tzinfo is not None

    def test_line_ids_are_unique_per_entry(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        for _ in range(5):
            ledger.append(factor=_make_factor(), result=_make_result())
        ids = [e.meta["ledger_line_id"] for e in ledger.read_all()]
        assert len(set(ids)) == 5

    def test_each_line_is_standalone_json(self, tmp_path: Path) -> None:
        """Each written line is independently parseable — a caller
        using jq/grep/pandas.read_json without the engine in hand
        gets usable data."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        ledger.append(factor=_make_factor(), result=_make_result())
        ledger.append(factor=_make_factor(), result=_make_result())
        raw = (tmp_path / "ledger.jsonl").read_text().strip().split("\n")
        assert len(raw) == 2
        for line in raw:
            obj = json.loads(line)  # each line parses standalone
            assert obj["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# count_experiments / count_finite_experiments / variance semantics
# ---------------------------------------------------------------------------


class TestCountAndVariance:
    def _write_sharpes(self, tmp_path: Path, sharpes: list[float]) -> Ledger:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        for i, s in enumerate(sharpes):
            ledger.append(
                factor=_make_factor(f"f_{i}"),
                result=_make_result(factor_id=f"f_{i:08d}", sharpe=s),
            )
        return ledger

    def test_count_experiments_is_literal_no_filter(self, tmp_path: Path) -> None:
        """count_experiments includes NaN-Sharpe entries. NaN results
        are still "trials" in the multiple-testing sense."""
        ledger = self._write_sharpes(tmp_path, [1.0, float("nan"), 2.0, float("nan"), 0.5])
        assert ledger.count_experiments() == 5

    def test_count_finite_experiments_drops_nan(self, tmp_path: Path) -> None:
        ledger = self._write_sharpes(tmp_path, [1.0, float("nan"), 2.0, float("nan"), 0.5])
        assert ledger.count_finite_experiments() == 3

    def test_count_finite_drops_inf(self, tmp_path: Path) -> None:
        """inf is also non-finite — filtered out."""
        ledger = self._write_sharpes(tmp_path, [1.0, float("inf"), 2.0, float("-inf"), 0.5])
        assert ledger.count_finite_experiments() == 3

    def test_variance_matches_sample_variance_on_finite(self, tmp_path: Path) -> None:
        """sharpe_variance_across_trials is ddof=1 sample variance,
        filtered to finite Sharpes. Match against a direct numpy
        computation on the same filtered set."""
        sharpes = [1.0, float("nan"), 2.0, 0.5, 1.5]
        ledger = self._write_sharpes(tmp_path, sharpes)
        expected = float(np.var([1.0, 2.0, 0.5, 1.5], ddof=1))
        assert ledger.sharpe_variance_across_trials() == pytest.approx(expected)

    def test_variance_nan_when_fewer_than_two_finite(self, tmp_path: Path) -> None:
        ledger = self._write_sharpes(tmp_path, [float("nan"), float("nan")])
        assert math.isnan(ledger.sharpe_variance_across_trials())

        ledger2 = self._write_sharpes(tmp_path, [1.0])
        assert math.isnan(ledger2.sharpe_variance_across_trials())

    def test_filter_fn_configurable_on_count(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        ledger.append(
            factor=_make_factor("keep_1"),
            result=_make_result(factor_id="f_0000000a", sharpe=1.0),
        )
        ledger.append(
            factor=_make_factor("drop_2"),
            result=_make_result(factor_id="f_0000000b", sharpe=2.0),
        )
        ledger.append(
            factor=_make_factor("keep_3"),
            result=_make_result(factor_id="f_0000000c", sharpe=1.5),
        )
        n = ledger.count_experiments(filter_fn=lambda e: e.factor.name.startswith("keep_"))
        assert n == 2

    def test_empty_ledger_stats(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        assert ledger.count_experiments() == 0
        assert ledger.count_finite_experiments() == 0
        assert math.isnan(ledger.sharpe_variance_across_trials())


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_current_schema_version_is_one(self) -> None:
        assert SCHEMA_VERSION == 1

    def test_unknown_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        # Write a line that declares schema_version=999 — from the future.
        bad_line = json.dumps({"schema_version": 999, "meta": {}, "factor": {}, "result": {}})
        path.write_text(bad_line + "\n")

        ledger = Ledger(path)
        with pytest.raises(Exception, match="schema_version"):
            list(ledger.read_all())

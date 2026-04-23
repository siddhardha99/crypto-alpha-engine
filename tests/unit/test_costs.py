"""Tests for the cost-rate calculators in backtest/costs.py.

This module is the leaf of the Phase 6 backtest layer — pure scalar
math (plus one Series-sum function for funding). No simulation, no
vectorbt. The higher-level simulation layer (commit 2) composes these
into the trade-by-trade cost application.

Principle 5 ("costs are mandatory") is enforced structurally: every
public function either returns a strictly positive rate for a valid
CostModel, or zero when a cost component is explicitly disabled
(``funding_applied=False``). There is no code path that silently
returns zero for fees or slippage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.costs import (
    CostModelSaturation,
    borrow_rate_per_period,
    compute_funding_charge,
    fee_rate,
    slippage_rate,
)
from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.types import CostModel

# ---------------------------------------------------------------------------
# Fee rate
# ---------------------------------------------------------------------------


class TestFeeRate:
    def test_default_taker_fee_matches_spec(self) -> None:
        """SPEC §8 default: taker_bps=10 → 0.001 (0.1%)."""
        assert fee_rate(CostModel()) == pytest.approx(0.001)

    def test_default_maker_fee_matches_spec(self) -> None:
        """SPEC §8 default: maker_bps=2 → 0.0002."""
        assert fee_rate(CostModel(), is_taker=False) == pytest.approx(0.0002)

    def test_taker_is_default(self) -> None:
        """Conservative default — taker is the worse side, so unless
        the caller explicitly opts into maker pricing, charge taker."""
        assert fee_rate(CostModel()) == fee_rate(CostModel(), is_taker=True)

    def test_custom_bps_scales_linearly(self) -> None:
        assert fee_rate(CostModel(taker_bps=50.0)) == pytest.approx(0.005)
        assert fee_rate(CostModel(maker_bps=1.0), is_taker=False) == pytest.approx(0.0001)

    def test_always_strictly_positive(self) -> None:
        """Principle 5 canary: any valid CostModel yields positive fees."""
        for taker, maker in [(0.1, 0.05), (10.0, 2.0), (100.0, 50.0)]:
            cm = CostModel(taker_bps=taker, maker_bps=maker)
            assert fee_rate(cm) > 0
            assert fee_rate(cm, is_taker=False) > 0


# ---------------------------------------------------------------------------
# Slippage rate (volume-based)
# ---------------------------------------------------------------------------


class TestSlippageRate:
    """SPEC §8 volume_based model:

    * trade / daily_volume < 1%  →  slippage = 0.05%       (floor)
    * 1% ≤ ratio ≤ 10%           →  quadratic rise to 1%   (cap at 10%)
    * ratio > 10%                →  saturated at 1%        (+ warning)
    """

    cm = CostModel()

    def test_small_trade_hits_floor(self) -> None:
        # 0.5% of daily volume → below 1% threshold → 0.05% flat.
        rate = slippage_rate(trade_notional=500.0, daily_volume=100_000.0, cost_model=self.cm)
        assert rate == pytest.approx(0.0005)

    def test_boundary_at_one_percent(self) -> None:
        # Exactly 1% of daily volume → still the floor.
        rate = slippage_rate(trade_notional=1_000.0, daily_volume=100_000.0, cost_model=self.cm)
        assert rate == pytest.approx(0.0005)

    def test_boundary_at_ten_percent(self) -> None:
        # Exactly 10% of daily volume → cap at 1%.
        rate = slippage_rate(trade_notional=10_000.0, daily_volume=100_000.0, cost_model=self.cm)
        assert rate == pytest.approx(0.01)

    def test_midpoint_quadratic(self) -> None:
        """At ratio = 5.5% (halfway from 1% to 10%), the quadratic says:
        0.0005 + 0.0095 · ((0.045 / 0.09)²) = 0.0005 + 0.0095 · 0.25 = 0.002875.
        """
        rate = slippage_rate(trade_notional=5_500.0, daily_volume=100_000.0, cost_model=self.cm)
        assert rate == pytest.approx(0.002875, rel=1e-6)

    def test_saturation_above_ten_percent_emits_warning(self) -> None:
        """Trading > 10% of daily volume is outside the modeled range.
        We cap at 1% and warn — silent saturation would understate cost.

        The warning is a proper ``CostModelSaturation`` subclass of
        ``UserWarning`` (not a plain UserWarning, not a log line), so
        callers can filter or escalate programmatically — e.g., set
        ``filterwarnings("error", category=CostModelSaturation)`` in
        property tests.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rate = slippage_rate(
                trade_notional=20_000.0, daily_volume=100_000.0, cost_model=self.cm
            )
        assert rate == pytest.approx(0.01)
        saturation_warnings = [w for w in caught if issubclass(w.category, CostModelSaturation)]
        assert len(saturation_warnings) == 1
        assert "slippage" in str(saturation_warnings[0].message).lower()

    def test_saturation_warning_is_escalatable_to_error(self) -> None:
        """A test that proves the warning can be turned into a hard
        failure by a caller — the whole point of using a subclass."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=CostModelSaturation)
            with pytest.raises(CostModelSaturation):
                slippage_rate(
                    trade_notional=20_000.0,
                    daily_volume=100_000.0,
                    cost_model=self.cm,
                )

    def test_monotonic_in_trade_size(self) -> None:
        """Holding volume fixed, larger trades have weakly-greater slippage
        at every step. A regression that broke the quadratic would trip this."""
        volume = 100_000.0
        sizes = np.linspace(500.0, 10_000.0, 50)
        rates = [slippage_rate(s, volume, self.cm) for s in sizes]
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_zero_or_negative_trade_raises(self) -> None:
        with pytest.raises(ConfigError, match="trade_notional"):
            slippage_rate(trade_notional=0.0, daily_volume=100_000.0, cost_model=self.cm)
        with pytest.raises(ConfigError, match="trade_notional"):
            slippage_rate(trade_notional=-1.0, daily_volume=100_000.0, cost_model=self.cm)

    def test_zero_or_negative_volume_raises(self) -> None:
        with pytest.raises(ConfigError, match="daily_volume"):
            slippage_rate(trade_notional=100.0, daily_volume=0.0, cost_model=self.cm)
        with pytest.raises(ConfigError, match="daily_volume"):
            slippage_rate(trade_notional=100.0, daily_volume=-5.0, cost_model=self.cm)

    def test_unknown_model_raises(self) -> None:
        """``slippage_model`` is a free-form string on CostModel; if a
        caller supplies an unknown value, fail loud — not silently
        default to volume_based."""
        bad = CostModel.__new__(CostModel)
        object.__setattr__(bad, "taker_bps", 10.0)
        object.__setattr__(bad, "maker_bps", 2.0)
        object.__setattr__(bad, "slippage_model", "magic_oracle")
        object.__setattr__(bad, "funding_applied", True)
        object.__setattr__(bad, "borrow_rate_bps", 20.0)
        with pytest.raises(ConfigError, match="slippage_model"):
            slippage_rate(trade_notional=100.0, daily_volume=100_000.0, cost_model=bad)


# ---------------------------------------------------------------------------
# Borrow rate
# ---------------------------------------------------------------------------


class TestBorrowRate:
    def test_converts_annual_bps_to_per_period(self) -> None:
        """20 bps annual, hourly bars (8760/yr) → 20/1e4 / 8760 ≈ 2.28e-7 per bar."""
        cm = CostModel(borrow_rate_bps=20.0)
        rate = borrow_rate_per_period(cm, periods_per_year=8760.0)
        assert rate == pytest.approx(0.002 / 8760.0, rel=1e-9)

    def test_daily_bars(self) -> None:
        """Same 20 bps over 365 bars → 0.002/365."""
        cm = CostModel(borrow_rate_bps=20.0)
        rate = borrow_rate_per_period(cm, periods_per_year=365.0)
        assert rate == pytest.approx(0.002 / 365.0, rel=1e-9)

    def test_four_hour_bars(self) -> None:
        """2190 four-hour bars per year is a real bar frequency. A bug
        in a rate-lookup table could make 8760 and 365 work while 2190
        silently broke; pin it separately."""
        cm = CostModel(borrow_rate_bps=20.0)
        rate = borrow_rate_per_period(cm, periods_per_year=2190.0)
        assert rate == pytest.approx(0.002 / 2190.0, rel=1e-9)

    def test_non_integer_periods_per_year(self) -> None:
        """``periods_per_year`` is typed as ``float`` — non-integer
        values (e.g., 365.25 for leap-year-aware daily) must still
        produce a sensible per-period rate."""
        cm = CostModel(borrow_rate_bps=20.0)
        rate = borrow_rate_per_period(cm, periods_per_year=365.25)
        assert rate == pytest.approx(0.002 / 365.25, rel=1e-9)

    @pytest.mark.parametrize("periods", [8760.0, 2190.0, 365.25, 52.0])
    def test_linear_round_trip(self, periods: float) -> None:
        """Linear (simple-interest) invariant: convert to per-bar, then
        multiply back by the bar count, get the original annual rate.

        This is the behavioural seal on "linear, not compound". If the
        function ever flipped to ``(1 + r)**(1/N) - 1``, this check
        would fire at every frequency simultaneously.
        """
        annual_bps = 20.0
        annual_fraction = annual_bps / 10_000.0
        cm = CostModel(borrow_rate_bps=annual_bps)
        per_period = borrow_rate_per_period(cm, periods_per_year=periods)
        reconstructed = per_period * periods
        assert reconstructed == pytest.approx(annual_fraction, rel=1e-12)

    def test_rejects_non_positive_periods_per_year(self) -> None:
        with pytest.raises(ConfigError, match="periods_per_year"):
            borrow_rate_per_period(CostModel(), periods_per_year=0.0)
        with pytest.raises(ConfigError, match="periods_per_year"):
            borrow_rate_per_period(CostModel(), periods_per_year=-1.0)


# ---------------------------------------------------------------------------
# Funding charge (series-sum)
# ---------------------------------------------------------------------------


def _aligned_series(position: list[float], funding: list[float]) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=len(position), freq="8h", tz="UTC")
    return pd.Series(position, index=idx), pd.Series(funding, index=idx)


class TestComputeFundingCharge:
    def test_sign_convention_single_bar_long_pays(self) -> None:
        """Sign convention pin, concrete scenario: one bar, +1 BTC long,
        funding rate +0.0001 → charge == +0.0001 (positive = paid out).

        Keeping this test explicit (single bar, single rate, one
        multiplicand on each side) pins the sign in the narrowest
        possible way. A regression that flipped the sign couldn't hide
        behind multi-bar sums."""
        idx = pd.date_range("2024-01-01", periods=1, freq="8h", tz="UTC")
        pos = pd.Series([1.0], index=idx)  # +1 BTC long
        fr = pd.Series([0.0001], index=idx)  # +0.01% funding
        charge = compute_funding_charge(pos, fr, CostModel())
        assert charge == pytest.approx(0.0001)
        assert charge > 0  # explicit: positive = cost to the long

    def test_sign_convention_single_bar_short_receives(self) -> None:
        """Symmetric pin: -1 BTC short at +0.0001 funding → charge ==
        -0.0001 (negative = received)."""
        idx = pd.date_range("2024-01-01", periods=1, freq="8h", tz="UTC")
        pos = pd.Series([-1.0], index=idx)  # -1 BTC short
        fr = pd.Series([0.0001], index=idx)  # +0.01% funding
        charge = compute_funding_charge(pos, fr, CostModel())
        assert charge == pytest.approx(-0.0001)
        assert charge < 0  # explicit: negative = receipt to the short

    def test_sign_convention_negative_funding_flips_signs(self) -> None:
        """When funding goes negative (shorts pay longs), signs flip:
        long at -0.0001 receives 0.0001, short at -0.0001 pays 0.0001.
        Pins the full 2x2 matrix of (long/short) × (funding+/funding-)."""
        idx = pd.date_range("2024-01-01", periods=1, freq="8h", tz="UTC")
        neg_fr = pd.Series([-0.0001], index=idx)
        long_pos = pd.Series([1.0], index=idx)
        short_pos = pd.Series([-1.0], index=idx)
        assert compute_funding_charge(long_pos, neg_fr, CostModel()) == pytest.approx(-0.0001)
        assert compute_funding_charge(short_pos, neg_fr, CostModel()) == pytest.approx(0.0001)

    def test_long_position_pays_positive_funding(self) -> None:
        """When funding is positive, longs pay shorts. Our sign convention:
        returned charge is a *cost* — positive number = money out."""
        pos, fr = _aligned_series([1.0, 1.0, 1.0], [0.0001, 0.0002, 0.0001])
        charge = compute_funding_charge(pos, fr, CostModel())
        assert charge == pytest.approx(1.0 * (0.0001 + 0.0002 + 0.0001))

    def test_short_position_earns_positive_funding(self) -> None:
        """Short pays negative funding (i.e., receives) when funding > 0."""
        pos, fr = _aligned_series([-1.0, -1.0], [0.0002, 0.0002])
        charge = compute_funding_charge(pos, fr, CostModel())
        assert charge == pytest.approx(-1.0 * 2 * 0.0002)

    def test_returns_zero_when_funding_disabled(self) -> None:
        """``CostModel(funding_applied=False)`` is the explicit opt-out for
        spot-only strategies. Must return 0.0 regardless of inputs — no
        partial application."""
        pos, fr = _aligned_series([1.0, -1.0, 0.5], [0.01, 0.01, 0.01])
        cm = CostModel(funding_applied=False)
        assert compute_funding_charge(pos, fr, cm) == 0.0

    def test_zero_position_no_charge(self) -> None:
        pos, fr = _aligned_series([0.0, 0.0, 0.0], [0.001, 0.001, 0.001])
        assert compute_funding_charge(pos, fr, CostModel()) == 0.0

    def test_index_mismatch_raises(self) -> None:
        """Position and funding must be on the same index — silent
        intersection would hide a bug where caller passed wrong series."""
        pos_idx = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
        fr_idx = pd.date_range("2024-01-02", periods=3, freq="8h", tz="UTC")
        pos = pd.Series([1.0, 1.0, 1.0], index=pos_idx)
        fr = pd.Series([0.0001, 0.0001, 0.0001], index=fr_idx)
        with pytest.raises(ConfigError, match="index"):
            compute_funding_charge(pos, fr, CostModel())

    def test_nan_funding_propagates_as_error(self) -> None:
        """NaN in funding_rate would silently zero out that period.
        Surface loudly instead — Phase 2 data layer shouldn't ship NaN
        funding, so a NaN here is a real data problem."""
        pos, fr = _aligned_series([1.0, 1.0], [0.0001, float("nan")])
        with pytest.raises(ConfigError, match="NaN"):
            compute_funding_charge(pos, fr, CostModel())

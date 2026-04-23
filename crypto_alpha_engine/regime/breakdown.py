"""Per-regime metric aggregation.

Given a returns series and one or more regime-label series (one per
timestamp), compute a chosen metric separately for each unique label.
The output is a ``dict[label, metric_value]``.

Signature accepts either a single :class:`pd.Series` of labels OR a
tuple of :class:`pd.Series`. The single-Series form is the Phase-5
use case (one regime dimension per breakdown). The tuple form is
documented as a supported extension: labels are joined by ``"×"``
(Cartesian product across dimensions, so ``bull × high_vol`` etc.)
for cross-regime metric rollups. That extension is the obvious
place for Phase 6/7 to hook in multi-dimensional breakdown
reporting without reworking the signature.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from crypto_alpha_engine.exceptions import ConfigError


def breakdown_by_regime(
    returns: pd.Series,
    labels: pd.Series | tuple[pd.Series, ...],
    metric_fn: Callable[[pd.Series], float],
) -> dict[str, float]:
    """Aggregate ``metric_fn`` over each regime label.

    Args:
        returns: The return series. Index must align with the label
            series (shared index intersection is used; non-overlapping
            timestamps are dropped silently).
        labels: Either one :class:`pd.Series` of string labels (one
            regime dimension) or a tuple of such Series (multi-
            dimensional: Cartesian product of label values becomes a
            single composite label joined by ``"×"``).
        metric_fn: A metric function like
            :func:`crypto_alpha_engine.backtest.metrics.sharpe`. Takes
            a ``pd.Series`` of returns, returns a ``float``.

    Returns:
        ``dict[label, float]``. Labels that are ``None`` or NaN in
        the input are excluded — they represent warmup-window
        ambiguity, not a real regime.

    Raises:
        ConfigError: If the tuple form has zero elements.

    Example:
        >>> # Single dimension:
        >>> breakdown_by_regime(returns, trend_labels, sharpe)
        # {"bull": 1.3, "bear": -0.4, "crab": 0.7}

        >>> # Two dimensions (Cartesian):
        >>> breakdown_by_regime(
        ...     returns, (trend_labels, vol_labels), sharpe
        ... )
        # {"bull × low_vol": 1.1, "bull × high_vol": 1.6, ...}
    """
    composite = _composite_labels(labels)

    # Align returns + labels on the shared index, drop rows where the
    # label is missing (warmup window), then group by label.
    idx = returns.index.intersection(composite.index)
    r = returns.loc[idx]
    lab = composite.loc[idx]
    mask = lab.notna()
    r = r[mask]
    lab = lab[mask]

    out: dict[str, float] = {}
    for label_value, group_returns in r.groupby(lab, observed=True):
        out[str(label_value)] = metric_fn(group_returns)
    return out


def _composite_labels(
    labels: pd.Series | tuple[pd.Series, ...],
) -> pd.Series:
    """Fold a single Series or a tuple of Series into one composite label Series.

    For a tuple, the output's label at timestamp t is
    ``"{labels[0][t]} × {labels[1][t]} × ..."`` when all parts are
    non-null; otherwise ``None``.
    """
    if isinstance(labels, pd.Series):
        return labels
    if not labels:
        raise ConfigError("labels tuple must contain at least one Series")
    if len(labels) == 1:
        return labels[0]

    # Align on index intersection across all label Series.
    common_idx = labels[0].index
    for s in labels[1:]:
        common_idx = common_idx.intersection(s.index)

    parts = [s.loc[common_idx] for s in labels]
    composite = pd.Series(index=common_idx, dtype=object)
    composite[:] = None
    # Only set a composite label where ALL parts are non-null.
    mask = pd.Series(True, index=common_idx)
    for p in parts:
        mask &= p.notna()
    # Build composite strings row-by-row for rows where mask is True.
    if mask.any():
        composite.loc[mask] = [" × ".join(str(p.loc[ts]) for p in parts) for ts in common_idx[mask]]
    return composite

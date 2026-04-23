"""Duplicate detection against the experiment ledger (SPEC §7, §11).

Structural-prefilter + behavioral-confirmation. The structural check
(Phase 4 :func:`ast_similarity`) is cheap; behavioral (Phase 4
:func:`behavioural_similarity`) requires compiling and evaluating
each prior against the candidate's features. We only pay the
behavioral cost for priors that pass the structural threshold.

Bounded work
------------

Three defenses against pathological input (from the Phase-7 review):

1. **Ordering.** Structural hits are processed in descending-similarity
   order; ties broken by newest-first (reverse ledger insertion order).
   First real duplicate found short-circuits the rest.
2. **Hard cap.** After ``hard_cap`` priors have been
   behaviorally-checked without a hit, stop and return
   ``DuplicateCheck(match=None, cap_exceeded=True, ...)``. Prevents a
   pathological new factor from triggering O(n) compile-and-evaluate
   cycles across a large ledger.
3. *(Phase 8 candidate)* Time budget — not implemented in commit 2.

Skipped priors don't count toward the cap: we didn't do the expensive
work, so the budget isn't consumed. The ``skipped_priors`` tuple on
the result records why each prior was skipped so the caller can
reason about the verdict.

Data-sufficiency precondition
-----------------------------

Phase 4's :func:`behavioural_similarity` bails to 0.0 on insufficient
non-NaN overlap. Two legitimately-similar factors with different
warmup periods (e.g., ``ts_mean(x, 20)`` vs ``ts_mean(x, 100)``) can
hit this bail path if the feature history is too short. To prevent
the false-novel bug, ``check_duplicate`` asserts at entry that
``min(len(s) for s in features.values()) >= 3 *
max_int_heuristic(candidate.root)``. Per-prior, if the prior's
max-int exceeds the same 3× threshold, the prior is added to
``skipped_priors`` rather than behaviorally checked — its score would
be meaningless anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.factor.ast import walk
from crypto_alpha_engine.factor.compiler import compile_factor
from crypto_alpha_engine.factor.complexity import unique_features
from crypto_alpha_engine.factor.similarity import (
    ast_similarity,
    behavioural_similarity,
)
from crypto_alpha_engine.ledger.ledger import Ledger, LedgerEntry
from crypto_alpha_engine.types import Factor, FactorNode

_DEFAULT_STRUCTURAL_THRESHOLD: float = 0.7
"""SPEC §7 structural-duplicate threshold."""

_DEFAULT_BEHAVIORAL_THRESHOLD: float = 0.9
"""SPEC §7 behavioral-duplicate threshold."""

_DEFAULT_HARD_CAP: int = 20
"""Default budget of behavioral comparisons per duplicate check."""

_DATA_SUFFICIENCY_MULTIPLIER: int = 3
"""Feature length must be at least this multiple of the factor's
max-int heuristic to count as "enough history for meaningful
behavioral comparison"."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateCheck:
    """Verdict of a :func:`check_duplicate` call.

    Three states:

    * ``match is None`` and ``cap_exceeded is False`` — candidate is
      novel, no duplicate found.
    * ``match is not None`` and ``cap_exceeded is False`` — candidate
      duplicates ``match`` (both thresholds cleared).
    * ``match is None`` and ``cap_exceeded is True`` — the check
      saturated before finding a hit; verdict is **indeterminate**.
      Caller must treat this as "unknown," not "novel." Typical
      remedies: prune the ledger, raise ``hard_cap``, or adopt a
      strategy ("raise" vs "skip") via ``on_duplicate`` in the
      engine wrapper.

    Attributes:
        match: The matched :class:`LedgerEntry`, or ``None``.
        structural_similarity: Highest structural score seen. Equal to
            the matched entry's score when ``match is not None``.
        behavioral_similarity: The matched entry's behavioral score, or
            the highest behavioral score observed among non-matches,
            or 0.0 if no behavioral check ran.
        cap_exceeded: True if the hard cap cut the search short.
        n_structural_hits: How many priors cleared the structural
            threshold (before cap).
        skipped_priors: ledger_line_ids of priors skipped for
            data-sufficiency reasons (prior's max-int exceeded the
            features-length budget).
    """

    match: LedgerEntry | None
    structural_similarity: float
    behavioral_similarity: float
    cap_exceeded: bool
    n_structural_hits: int
    skipped_priors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_duplicate(
    *,
    candidate: Factor,
    candidate_features: dict[str, pd.Series],
    ledger: Ledger,
    structural_threshold: float = _DEFAULT_STRUCTURAL_THRESHOLD,
    behavioral_threshold: float = _DEFAULT_BEHAVIORAL_THRESHOLD,
    hard_cap: int = _DEFAULT_HARD_CAP,
) -> DuplicateCheck:
    """Check whether ``candidate`` duplicates any prior entry in ``ledger``.

    Two-stage check:

    1. :func:`ast_similarity` against every prior's factor AST. Priors
       clearing ``structural_threshold`` enter stage 2.
    2. For each structural hit (descending similarity, newest-first on
       ties): compile the prior, evaluate on ``candidate_features``,
       compute :func:`behavioural_similarity`. First hit with score
       ``>= behavioral_threshold`` short-circuits the rest.

    Args:
        candidate: The factor being checked.
        candidate_features: The features dict used for behavioral
            comparison. Must have enough history for the candidate's
            max-window kwarg (see data-sufficiency precondition in
            module docstring).
        ledger: The ledger to compare against.
        structural_threshold: AST-similarity cutoff for stage 1.
            Default ``0.7`` per SPEC §7.
        behavioral_threshold: Behavioral-correlation cutoff for stage
            2. Default ``0.9`` per SPEC §7.
        hard_cap: Max number of priors behaviorally checked before
            giving up with ``cap_exceeded=True``.

    Returns:
        :class:`DuplicateCheck`.

    Raises:
        ConfigError: If ``candidate_features`` is empty or too short
            for the candidate's warmup.
    """
    if not candidate_features:
        raise ConfigError("check_duplicate: candidate_features is empty")

    feature_len = min(len(s) for s in candidate_features.values())
    candidate_warmup = _max_int_heuristic(candidate.root)
    required_bars = _DATA_SUFFICIENCY_MULTIPLIER * candidate_warmup
    if feature_len < required_bars:
        raise ConfigError(
            f"check_duplicate: insufficient feature history — "
            f"candidate's max-window heuristic is {candidate_warmup}, "
            f"requires at least {_DATA_SUFFICIENCY_MULTIPLIER}× = "
            f"{required_bars} bars, features provide only {feature_len}. "
            f"Behavioral similarity against a prior with a different "
            f"warmup would be unreliable on this short a window."
        )

    # Load + rank priors. Enumerate to track ledger order for tie-breaking.
    priors: list[tuple[int, LedgerEntry, float]] = []
    for position, entry in enumerate(ledger.read_all()):
        score = ast_similarity(candidate.root, entry.factor.root)
        priors.append((position, entry, score))

    # Filter to structural hits.
    structural_hits = [
        (pos, entry, score) for (pos, entry, score) in priors if score >= structural_threshold
    ]
    # Sort: descending similarity; ties broken by newest-first
    # (higher position = more recent).
    structural_hits.sort(key=lambda t: (-t[2], -t[0]))

    n_structural_hits = len(structural_hits)
    if n_structural_hits == 0:
        max_observed_structural = max((score for (_, _, score) in priors), default=0.0)
        return DuplicateCheck(
            match=None,
            structural_similarity=max_observed_structural,
            behavioral_similarity=0.0,
            cap_exceeded=False,
            n_structural_hits=0,
            skipped_priors=(),
        )

    compiled_candidate = compile_factor(candidate)
    candidate_output = compiled_candidate(candidate_features)

    skipped: list[str] = []
    behavioral_checks_done = 0
    highest_behavioral = 0.0

    for _, entry, struct_score in structural_hits:
        if behavioral_checks_done >= hard_cap:
            return DuplicateCheck(
                match=None,
                structural_similarity=structural_hits[0][2],
                behavioral_similarity=highest_behavioral,
                cap_exceeded=True,
                n_structural_hits=n_structural_hits,
                skipped_priors=tuple(skipped),
            )

        # Data-sufficiency for THIS prior.
        prior_warmup = _max_int_heuristic(entry.factor.root)
        if feature_len < _DATA_SUFFICIENCY_MULTIPLIER * prior_warmup:
            skipped.append(entry.meta["ledger_line_id"])
            continue

        # Feature-dependency check: prior may reference feature keys
        # that aren't in candidate_features. If so, we can't evaluate
        # prior fairly — skip it.
        prior_feature_keys = unique_features(entry.factor.root)
        if not prior_feature_keys.issubset(candidate_features.keys()):
            skipped.append(entry.meta["ledger_line_id"])
            continue

        try:
            compiled_prior = compile_factor(entry.factor)
            prior_output = compiled_prior(candidate_features)
        except Exception:
            # Prior can't be compiled or evaluated on current features —
            # skip, don't pollute the cap.
            skipped.append(entry.meta["ledger_line_id"])
            continue

        behav = behavioural_similarity(candidate_output, prior_output)
        behavioral_checks_done += 1
        highest_behavioral = max(highest_behavioral, behav)

        if behav >= behavioral_threshold:
            return DuplicateCheck(
                match=entry,
                structural_similarity=struct_score,
                behavioral_similarity=behav,
                cap_exceeded=False,
                n_structural_hits=n_structural_hits,
                skipped_priors=tuple(skipped),
            )

    # No behavioral hit; search completed within cap.
    return DuplicateCheck(
        match=None,
        structural_similarity=structural_hits[0][2],
        behavioral_similarity=highest_behavioral,
        cap_exceeded=False,
        n_structural_hits=n_structural_hits,
        skipped_priors=tuple(skipped),
    )


# ---------------------------------------------------------------------------
# Max-int heuristic
# ---------------------------------------------------------------------------


def _max_int_heuristic(root: FactorNode) -> int:
    """Return the largest integer arg or kwarg anywhere in the AST.

    Heuristic: for the standard operator library, the largest int is
    the factor's warmup-window proxy (ts_mean(x, 20) has warmup 20;
    ts_zscore(ts_mean(x, 20), 100) has warmup 100). Accurate for
    every operator in Phase 3-6.

    Limitation: the heuristic does **not compose**. A deeply-nested
    factor like ``ts_mean(ts_mean(x, 20), 30)`` has true effective
    warmup of 20 + 30 - 1 = 49 bars, but this function returns 30
    (the max int, not the sum). Callers relying on the heuristic for
    data-sufficiency sizing buy slack via the 3× multiplier in
    :func:`check_duplicate`, which covers moderate nesting but can
    be undersized for pathologically deep factors. Workaround for
    now: pass a longer feature history. A proper compound-warmup
    tracker — walking the tree and summing per-operator warmup
    contributions — is a Phase 8+ addition if nested factors prove
    common in practice.

    Returns 0 if no ints are present (factor has no windowed ops →
    no meaningful warmup).

    Booleans are excluded even though ``isinstance(True, int)`` is
    True in Python — a common trap when introspecting int args.
    The broader pattern: ``type(v) is int`` or an explicit
    ``isinstance(v, int) and not isinstance(v, bool)`` check is
    required any time int-ness is semantically load-bearing.
    """
    max_int = 0
    for node in walk(root):
        for arg in node.args:
            # isinstance(True, int) is True in Python; exclude booleans.
            if isinstance(arg, int) and not isinstance(arg, bool):
                max_int = max(max_int, arg)
        for v in node.kwargs.values():
            if isinstance(v, int) and not isinstance(v, bool):
                max_int = max(max_int, v)
    return max_int

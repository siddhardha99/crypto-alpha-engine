# Contributing

Thanks for your interest in `crypto-alpha-engine`. Before opening a PR,
please read [`SPEC.md`](SPEC.md) (what the engine must do),
[`docs/methodology.md`](docs/methodology.md) (design rationale), and
[`CLAUDE.md`](CLAUDE.md) (engineering conventions).

The test suite is the first reviewer. PRs that don't pass the full
gauntlet — `ruff`, `black --check`, `mypy --strict`, `pytest` — won't
get human review; fix the suite first. CI runs the gauntlet on every
PR automatically, but running it locally before submission is strongly
recommended.

## The gauntlet

```bash
uv run pytest \
  && uv run ruff check . \
  && uv run black --check . \
  && uv run mypy crypto_alpha_engine tests scripts
```

Every PR must pass this exact command. If any piece fails, the PR is
not reviewable.

## Architectural non-negotiables

These aren't style preferences — they're invariants the engine's
security model depends on. PRs that weaken any of them will be
closed regardless of the problem they solve.

1. **Principle 2 (causality is sacred).** Never disable, bypass, or
   weaken a causality check to make a test pass. If a test reveals a
   lookahead, fix the operator or factor — not the test. If a
   causality canary flags your code, read the message and fix the
   root cause.
2. **Principle 1 (sealed engine).** `BacktestResult` exposes aggregate
   scalars only. Never add a field that returns an equity curve,
   per-bar returns, trade timestamps, or any time-series data. If you
   think you need one, flag it in an issue first — the answer is
   probably "log it, don't return it."
3. **Principle 5 (costs are mandatory).** `CostModel` must remain a
   required kwarg on every simulation path. No default-zero overloads,
   no `apply_costs=False` parameter, no exceptions.

## Adding a new operator

The operator registry is the largest contribution surface. Rules:

1. **The kernel is a pure function on `pd.Series`.** No I/O, no
   mutable state, no globals that persist across calls.
2. **Register with `causal_safe=True`** unless you're deliberately
   shipping a research-purpose acausal operator, in which case
   document explicitly why the registry's default was overridden.
3. **Write a per-operator test file** covering: normal path,
   insufficient-data edge case, NaN propagation, and any operator-
   specific failure mode.
4. **Add the operator to the catalogue.** New-operator PRs **must
   include both** the per-operator causality test AND the catalogue
   entry in
   [`tests/unit/test_operators_causality.py`](tests/unit/test_operators_causality.py).
   The coverage canary walks the registry and asserts every
   registered operator has a catalogue entry; without the entry the
   canary fails. **PRs that pass other tests but fail the catalogue
   canary will not be merged** — it exists specifically to prevent
   operators slipping past causality testing.

## Adding a new data source

1. **Implement the `DataSource` Protocol** from
   [`docs/adding_custom_sources.md`](docs/adding_custom_sources.md).
   The template there is the canonical reference.
2. **Conform to the schema for your `DataType`.** Pandera schemas in
   [`crypto_alpha_engine/data/schemas.py`](crypto_alpha_engine/data/schemas.py)
   are strict; loading violates-schema data raises
   `DataSchemaViolation` at load time.
3. **Include an end-to-end unit test** that goes
   `parse → compile → evaluate` using a factor referencing the new
   source's features. The test should use a small in-repo fixture
   (a handful of rows committed to `tests/fixtures/`), not live API
   calls.

## Commit style

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`,
  `chore:`. Use one prefix per commit; split mixed-concern changes.
- Commit messages explain the **why**, not just the what.
- Small commits are better than large ones — <300 lines of diff when
  possible. Each commit must pass the gauntlet independently.

## PR description

Include:

- **What** — one sentence on what the change does.
- **Why** — one short paragraph on the motivation (ticket, issue,
  research finding).
- **How tested** — the specific tests added or modified, plus "full
  gauntlet green locally."
- **Breaking changes** — yes/no, and what if yes.

## What doesn't get reviewed

- PRs that fail the gauntlet.
- PRs adding an operator without the catalogue entry.
- PRs weakening any of the three architectural non-negotiables.
- PRs adding dependencies without justifying why the existing set
  doesn't cover the need.

## Everything else

When in doubt, open an issue describing the change first. The cost
of a 10-line issue discussion is much lower than the cost of a
200-line PR that's going in the wrong direction.

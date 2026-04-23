# CLAUDE.md — Operating Instructions for Claude Code

This is your operating manual. Read it fully before writing any code.

---

## Project Context

You are building `crypto-alpha-engine` — an open-source crypto quantitative research engine. The full specification is in `SPEC.md`. This file tells you HOW to work; SPEC.md tells you WHAT to build.

---

## Your Role

You are a senior Python engineer implementing a production-grade backtesting library. You care about:
- Correctness over speed
- Explicit over implicit
- Tests over assumptions
- Clear error messages over silent failures

You are NOT building a trading bot. You are NOT giving financial advice. You are building rigorous research infrastructure.

---

## Rules of Engagement

### Rule 1: Read first, code second
Before implementing any module, read:
1. The relevant section of SPEC.md
2. The test file for that module (in `tests/`)
3. Any related code that already exists

### Rule 2: Test-driven, non-negotiable
For every module:
1. Write the failing tests first (based on SPEC.md section 13)
2. Run tests, confirm they fail
3. Implement the minimum code to pass
4. Refactor while keeping tests green
5. Commit with passing tests

### Rule 3: Follow the implementation order
SPEC.md section 14 defines the exact order. Do not skip ahead. Each phase's exit criteria must be met before the next phase starts.

### Rule 4: The engine is sealed — forever
The user (or another AI) must never have a code path that exposes:
- Raw equity curves
- Trade-level timestamps
- Per-trade PnL
- Daily returns series

If you find yourself adding a `return_equity_curve=True` parameter, stop. That's not a feature, that's a bug.

### Rule 5: Causality is sacred
If you ever write `.shift(-N)` with positive N in an operator, you have made a mistake. Stop.

If a test fails because an operator can look ahead, do NOT modify the test. Fix the operator.

### Rule 6: Ask before adding dependencies
The dependency list in SPEC.md is deliberate. If you want to add a package, first check if existing dependencies cover it. If you truly need a new one, document why in the commit.

### Rule 7: No silent fallbacks
If something can fail, it must fail loudly with a specific exception type and a clear message. No `try: ... except: pass`. No `return None` where failure is plausible.

### Rule 8: Type hints are mandatory
`mypy --strict` must pass. Every function has type hints on all parameters and return value.

### Rule 9: Docstrings are part of the feature
Every public function needs a Google-style docstring with:
- One-line summary
- Args with types and meaning
- Returns with type and meaning
- Raises (which exceptions, when)
- Example (at least one)

### Rule 10: Commit discipline
- Small commits (< 300 lines of diff when possible)
- Conventional commit format: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- Each commit passes tests on its own
- Commit message explains the "why", not just the "what"

---

## Workflow for Each Module

```
1. Read SPEC.md section for this module
2. Create test file with failing tests from SPEC.md section 13
3. Run: pytest tests/unit/test_<module>.py -v
   → Confirm tests fail for the right reason
4. Implement minimum code to pass first test
5. Run tests, iterate
6. When all tests pass, run full suite: pytest
7. Run type checker: mypy --strict crypto_alpha_engine/
8. Run linter: ruff check crypto_alpha_engine/ tests/
9. Run formatter: black crypto_alpha_engine/ tests/
10. Commit with message: "feat(<module>): <what it does>"
11. Update docs if this is a public API
```

---

## When You Get Stuck

Priority order for resolving ambiguity:

1. **Re-read SPEC.md** — the answer is usually there
2. **Look at an analogous module** — consistency matters
3. **Check methodology.md** — design rationale lives there
4. **Ask the human** — do not guess silently on architectural decisions

For small local decisions (variable names, internal function structure), use your judgment and document in the docstring.

For big decisions (should this be a class or a function, should this fail or warn), ask.

---

## Common Pitfalls to Avoid

### Pitfall 1: Returning too much from the engine
If you're tempted to return "just the daily returns for debugging," you're breaking Principle 1 (sealed engine). Add it to logs, not to the return value.

### Pitfall 2: Using `.iloc` with positive offsets in operators
Any positive offset in `.iloc` is a potential look-ahead. Use `.shift(positive_lag)` instead — shift with positive integer looks BACKWARD in time.

### Pitfall 3: Assuming UTC
Always be explicit. `pd.Timestamp` should always have `tz='UTC'`. Reject any timestamp that isn't timezone-aware.

### Pitfall 4: Mutable default arguments
```python
# WRONG
def foo(x, history=[]): ...

# RIGHT
def foo(x, history=None):
    if history is None:
        history = []
```

### Pitfall 5: "Just this once" exceptions to the architecture
If you find yourself writing "this is a special case that needs to bypass X," stop and ask. Special cases are how systems rot.

### Pitfall 6: Skipping tests because "this is trivial"
Nothing is trivial in a rigor-focused project. Trivial-looking bugs in quant systems eat years of returns.

### Pitfall 7: Generic exceptions
```python
# WRONG
raise Exception("something went wrong")
raise ValueError("bad input")

# RIGHT
raise LookAheadDetected(f"Factor {factor.name} references index {t+k} at time {t}")
raise DataSchemaViolation(f"Column 'close' has {n_negative} negative values in {symbol}")
```

### Pitfall 8: Premature optimization
Write clear code first. Profile before optimizing. Most of this code runs in seconds, not milliseconds.

---

## Testing Discipline

### When adding a feature, also add:
- At least one positive test (happy path)
- At least one negative test (what should fail)
- One edge case test (empty input, single row, NaN handling)

### Test naming
```python
def test_<what>_<condition>_<expected>():
    # test_ts_mean_with_insufficient_data_returns_nan()
    ...
```

### Test organization
- One test file per module
- Fixtures in `conftest.py`
- Integration tests in `tests/integration/`
- Property tests in `tests/property/`

### Fixtures use real data
Do not use synthetic `np.random.randn(1000)` for behavior tests. Use the sample BTC/ETH parquet files in `tests/fixtures/`. Real data catches real bugs.

---

## Documentation Discipline

### For every public function/class, write docstring as you write the code, not later.

### Update README.md when:
- You add a new user-facing feature
- You change the public API
- You add a new dependency

### Update methodology.md when:
- You make a design decision that's non-obvious
- You choose one algorithm over another
- You set a threshold or default value that has a rationale

---

## Git Workflow

### Branch strategy
- `main` — always passes CI, always deployable
- `feat/<name>` — feature branches
- `fix/<name>` — bug fix branches

### Before opening a PR
1. All tests pass
2. Coverage ≥ 90% for modified modules
3. `mypy --strict` passes
4. `ruff check` passes
5. `black` has been run
6. Docstrings updated
7. README or methodology.md updated if needed

### PR description template
```
## What
<one sentence>

## Why
<one paragraph>

## How tested
<list of tests that cover this>

## Breaking changes
<yes/no, and what>
```

---

## Working with SPEC.md

SPEC.md is the source of truth. If you think it's wrong, do not silently "fix" it in code. Instead:
1. Open an issue describing the discrepancy
2. Wait for the human to confirm the change
3. Update SPEC.md and the code together

Never let code and spec drift out of sync.

---

## Performance Targets (for reference, not optimization targets)

- Loading 5 years of 1h BTC data: < 1 second
- Running one walk-forward backtest: < 5 seconds
- Computing AST similarity for 1000 factors: < 100ms
- Appending to ledger: < 10ms

If you hit these naturally, great. Don't prematurely optimize to beat them.

---

## Security Considerations

### Things you will NEVER do
- Execute arbitrary user-supplied Python code (only parse factor JSON → AST → compiled operators)
- Make network calls from inside the backtest engine (data is pre-downloaded)
- Read or write files outside the project directory
- Log sensitive information (API keys, personal data)

### Things to be paranoid about
- User-supplied factor JSON could be maliciously crafted → validate schema strictly
- Ledger file integrity → use append-only open mode
- Data file provenance → hash every parquet at load time, log the hash in BacktestResult

---

## Your Priorities (in order)

1. **Correctness** — the engine must not lie about strategy performance
2. **Safety** — no cheating possible, by construction
3. **Clarity** — future-you and other contributors must understand this code
4. **Completeness** — every piece from SPEC.md, nothing more, nothing less
5. **Performance** — fast enough, not faster

If these ever conflict, higher-priority wins.

---

## First Session Plan

When you start work, do these in order:

1. `git init` the repo, create the basic structure from SPEC.md section 4
2. Write `pyproject.toml` with all dependencies
3. Write `.github/workflows/test.yml` for CI
4. Write `tests/conftest.py` with shared fixtures
5. Implement `types.py` (all dataclasses)
6. Implement `exceptions.py` (all custom exceptions)
7. Run empty test suite to verify CI works
8. First commit: `feat: scaffold project structure and base types`

Then proceed to Phase 2 (Data Layer) per SPEC.md section 14.

---

## Closing Instructions

You are not writing a blog post demo. You are writing infrastructure that will be publicly audited by the quant community. Every choice you make will be examined. This is a privilege — most code never gets that scrutiny — so build accordingly.

Be rigorous. Be explicit. Be honest about edge cases. When you complete Phase 1, someone reading the README should think: "Finally, a crypto research engine that takes causality seriously."

That's the standard.

---

*End of operating instructions.*

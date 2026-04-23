# Contributing

Thanks for your interest in `crypto-alpha-engine`. Before contributing, please
read [`SPEC.md`](SPEC.md) (what the engine must do) and
[`CLAUDE.md`](CLAUDE.md) (engineering conventions and non-negotiables).

This file will be expanded in Phase 8 per SPEC §14. For now:

- All Python tooling runs through `uv` — no raw `pip`.
- Every change must pass `uv run pytest`, `uv run ruff check .`,
  `uv run black --check .`, and `uv run mypy crypto_alpha_engine tests`.
- Commits follow conventional-commit style (`feat:`, `fix:`, `test:`,
  `docs:`, `refactor:`, `chore:`).
- Never weaken causality or seal-the-engine guarantees to make a test pass.
  If a test reveals a look-ahead, fix the operator, not the test.

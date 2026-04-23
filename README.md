# crypto-alpha-engine

A sealed, cheat-proof backtesting engine for crypto quantitative research.

## Status

Under active construction. See [`SPEC.md`](SPEC.md) for the full build
specification and [`CLAUDE.md`](CLAUDE.md) for engineering conventions.

This README will be fleshed out in Phase 8 (Polish & Docs) per SPEC §14.
For now, the quickstart, methodology, and API docs are intentionally absent —
do not treat the current state as a public release.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Development

```bash
uv sync --extra dev       # create venv, install runtime + dev deps
uv run pytest             # run the test suite
uv run ruff check .       # lint
uv run black --check .    # formatter check
uv run mypy crypto_alpha_engine tests   # strict type check
```

## License

MIT. See [`LICENSE`](LICENSE).

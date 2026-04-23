"""In-process registry of data sources.

A source must be registered before the loader or any orchestrator can
reach it. Built-in sources auto-register at
``import crypto_alpha_engine.data.sources``. Custom sources register
programmatically at startup of the user's script.

The registry is a process-local singleton. It does NOT persist across
Python processes — downstream tools that want persistent source
configuration read their own config files and call
:func:`register_source` at startup.
"""

from __future__ import annotations

from collections.abc import Iterable

from crypto_alpha_engine.data.protocol import DataSource, DataType
from crypto_alpha_engine.exceptions import ConfigError

_SOURCES: dict[str, DataSource] = {}


def register_source(source: DataSource) -> None:
    """Add ``source`` to the registry.

    Args:
        source: An object implementing :class:`DataSource`.

    Raises:
        ConfigError: If another source with the same ``name`` is already
            registered. The registry refuses shadowing rather than
            letting late imports silently mask earlier ones.
    """
    if source.name in _SOURCES:
        existing = _SOURCES[source.name]
        raise ConfigError(
            f"cannot register source {source.name!r}: another source "
            f"with that name is already registered "
            f"(existing data_type={existing.data_type.value!r}, "
            f"new data_type={source.data_type.value!r})"
        )
    _SOURCES[source.name] = source


def unregister_source(name: str) -> None:
    """Remove the source named ``name``, if present.

    Only exposed for test isolation. Production code should never call
    this; re-running a script after adding sources should be a no-op.
    """
    _SOURCES.pop(name, None)


def get_source(name: str) -> DataSource:
    """Return the source registered under ``name``.

    Raises:
        ConfigError: If no source is registered under that name.
    """
    try:
        return _SOURCES[name]
    except KeyError as err:
        raise ConfigError(
            f"no source registered under name {name!r}; " f"known: {sorted(_SOURCES)!r}"
        ) from err


def list_sources_for(
    data_type: DataType,
    symbol: str | None = None,
) -> list[DataSource]:
    """Return every source whose ``data_type`` matches, optionally filtered by symbol.

    Order of the returned list is registration order — the first source
    registered for a ``(data_type, symbol)`` pair is the default.

    Args:
        data_type: The data type to filter on.
        symbol: If given, only sources whose ``symbols`` list contains
            this symbol are returned.

    Returns:
        A list (possibly empty). Callers should not mutate it.
    """
    out: list[DataSource] = []
    for source in _SOURCES.values():
        if source.data_type != data_type:
            continue
        if symbol is not None and symbol not in source.symbols:
            continue
        out.append(source)
    return out


def default_for(data_type: DataType, symbol: str) -> DataSource:
    """Return the first-registered source for the given ``(data_type, symbol)``.

    Raises:
        ConfigError: If no source can serve this combination.
    """
    candidates = list_sources_for(data_type, symbol)
    if not candidates:
        raise ConfigError(
            f"no registered source for data_type={data_type.value!r} " f"symbol={symbol!r}"
        )
    return candidates[0]


def iter_all_sources() -> Iterable[DataSource]:
    """Iterate every registered source. For diagnostic / dump-config use."""
    return iter(_SOURCES.values())


def _reset_for_tests() -> None:
    """Clear the registry. Tests only — production code never calls this."""
    _SOURCES.clear()

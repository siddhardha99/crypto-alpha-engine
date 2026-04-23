"""Smoke tests for the package itself — it imports and has a version string."""

from __future__ import annotations

import re

import crypto_alpha_engine


def test_package_has_version_string() -> None:
    """Package exposes a ``__version__`` attribute in PEP 440 form."""
    version = crypto_alpha_engine.__version__
    assert isinstance(version, str)
    # Loose PEP 440 sanity: at least major.minor.patch, digits and dots.
    assert re.match(r"^\d+\.\d+\.\d+", version), f"unexpected version: {version!r}"


def test_package_is_importable_as_module() -> None:
    """The package imports without side effects that raise."""
    import importlib

    mod = importlib.import_module("crypto_alpha_engine")
    assert mod.__name__ == "crypto_alpha_engine"

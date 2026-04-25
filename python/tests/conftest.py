"""Pytest configuration for air_rs test suite.

Inserts the `python/` source tree into sys.path so tests can be run with
any Python interpreter (system, venv, or via `pytest` directly) without
needing PYTHONPATH to be set manually.

Tier 1 (pure-Python utils) always run.
Tier 2 (extension smoke) skip when the Rust .so is not compiled.
Tier 3 (integration) skip unless AIR_RS_TEST_MODEL env var points to a GGUF.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make `python/` importable regardless of how pytest was invoked
# ---------------------------------------------------------------------------

_PYTHON_SRC = Path(__file__).parent.parent.parent / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_model: test requires a real GGUF model file on disk",
    )
    config.addinivalue_line(
        "markers",
        "requires_extension: test requires the compiled Rust extension",
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def extension_available() -> bool:
    """True if the compiled _air_rs native extension can be imported."""
    try:
        importlib.import_module("air_rs._air_rs")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.fixture(scope="session")
def air_rs(extension_available: bool):  # type: ignore[return]
    """Import air_rs or skip the test if extension not built."""
    if not extension_available:
        pytest.skip(
            "Rust extension not compiled — run: maturin develop --features python"
        )
    import air_rs as m
    return m

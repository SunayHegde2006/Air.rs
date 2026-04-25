"""Air.rs — high-performance LLM inference engine.

This package wraps the Rust Air.rs inference engine via PyO3.
Pure-Python helpers (utils.format_chat, utils.count_tokens_approx) work
without the compiled extension. Engine/GenerateConfig/GbnfConstraint/Metrics
require the compiled `_air_rs` native extension:

    pip install air-rs         # from PyPI
    maturin develop --features python   # dev build from source

Quick start
-----------
>>> import air_rs
>>> engine = air_rs.Engine.from_gguf("model.gguf")
>>> print(engine.generate("Explain attention in one sentence."))
"""

from __future__ import annotations

# Pure-Python submodule — always importable
from air_rs import utils  # noqa: F401

# Native extension — lazy import so `utils` works even without the .so
try:
    from air_rs._air_rs import (  # noqa: F401
        Engine as Engine,
        GenerateConfig as GenerateConfig,
        GbnfConstraint as GbnfConstraint,
        Metrics as Metrics,
        __version__ as __version__,
    )
    _EXTENSION_LOADED = True
except ImportError:
    _EXTENSION_LOADED = False
    __version__ = "0.0.0+unbuilt"

    class _NotBuilt:
        """Placeholder raised when the Rust extension has not been compiled."""

        _name: str

        def __init_subclass__(cls, name: str = "", **kw: object) -> None:
            cls._name = name

        def __call__(self, *a: object, **kw: object) -> None:  # type: ignore[override]
            raise ImportError(
                f"air_rs.{self._name} requires the compiled Rust extension.\n"
                "Run: maturin develop --features python\n"
                "Or:  pip install air-rs"
            )

        @classmethod
        def __class_getitem__(cls, item: object) -> object:
            return cls

    class Engine(_NotBuilt, name="Engine"):  # type: ignore[no-redef]
        pass

    class GenerateConfig(_NotBuilt, name="GenerateConfig"):  # type: ignore[no-redef]
        pass

    class GbnfConstraint(_NotBuilt, name="GbnfConstraint"):  # type: ignore[no-redef]
        pass

    class Metrics(_NotBuilt, name="Metrics"):  # type: ignore[no-redef]
        pass


__all__ = [
    "Engine",
    "GenerateConfig",
    "GbnfConstraint",
    "Metrics",
    "utils",
    "__version__",
]

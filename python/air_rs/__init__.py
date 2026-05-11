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

Async streaming (issue #6)
--------------------------
>>> import asyncio
>>> async def main():
...     async for token in air_rs.astream(engine, "Once upon a time"):
...         print(token, end="", flush=True)
>>> asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

# Pure-Python submodule — always importable
from air_rs import utils  # noqa: F401

if TYPE_CHECKING:
    from air_rs._air_rs import (
        Engine,
        GbnfConstraint,
        GenerateConfig,
        Metrics,
        TokenChannel,
    )

# Native extension — lazy import so `utils` works even without the .so
try:
    from air_rs._air_rs import (  # isort: skip
        Engine as Engine,
        GbnfConstraint as GbnfConstraint,
        GenerateConfig as GenerateConfig,
        Metrics as Metrics,
        TokenChannel as TokenChannel,
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

        def __call__(self, *a: object, **kw: object) -> None:
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

    class TokenChannel(_NotBuilt, name="TokenChannel"):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# astream — native async generator for token-by-token streaming (issue #6)
# ---------------------------------------------------------------------------

# Module-level thread pool for recv_sync calls.
# Bounded pool (max 4 workers) avoids spawning one thread per
# concurrent astream call while still allowing multiple parallel streams.
_STREAM_EXECUTOR: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _STREAM_EXECUTOR
    if _STREAM_EXECUTOR is None:
        _STREAM_EXECUTOR = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="air_rs_stream",
        )
    return _STREAM_EXECUTOR


async def astream(
    engine: Engine,
    prompt: str,
    config: GenerateConfig | None = None,
    *,
    executor: ThreadPoolExecutor | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields decoded tokens one at a time.

    The Rust generation loop runs synchronously on a thread-pool thread.
    Each ``await`` yields control back to the event loop between tokens,
    keeping the loop fully responsive (no GIL holds, no event-loop blocking).

    Parameters
    ----------
    engine : Engine
        A loaded ``air_rs.Engine`` instance.
    prompt : str
        The input text prompt.
    config : GenerateConfig | None
        Optional per-call sampling config.
    executor : ThreadPoolExecutor | None
        Thread-pool executor for ``recv_sync`` calls.
        Defaults to the module-level shared pool (max 4 threads).

    Yields
    ------
    str
        Each decoded token string as it is produced.

    Examples
    --------
    >>> async for token in air_rs.astream(engine, "Tell me a story"):
    ...     print(token, end="", flush=True)

    With custom config:
    >>> cfg = air_rs.GenerateConfig(temperature=0.0, max_tokens=128)
    >>> async for token in air_rs.astream(engine, "2 + 2 =", cfg):
    ...     print(token, end="", flush=True)
    """
    if not _EXTENSION_LOADED:
        raise ImportError(
            "air_rs.astream requires the compiled Rust extension.\n"
            "Run: maturin develop --features python"
        )

    pool = executor or _get_executor()
    loop = asyncio.get_event_loop()

    # Obtain a live TokenChannel — generation starts immediately on this call.
    # generate_stream_inner fills the channel token-by-token; _stream_channel
    # returns once generation is complete (tokens buffered in the channel).
    channel: TokenChannel = await loop.run_in_executor(
        pool,
        lambda: engine._stream_channel(prompt, config),
    )

    # Drain channel: each recv_sync() blocks the pool thread (not the loop)
    # until the next token arrives or the stream is exhausted.
    while True:
        token: str | None = await loop.run_in_executor(
            pool,
            channel.recv_sync,
        )
        if token is None:
            return
        yield token


# ---------------------------------------------------------------------------
# shutdown helper — call at program exit to clean up the thread pool
# ---------------------------------------------------------------------------

def shutdown_stream_executor(wait: bool = True) -> None:
    """Shut down the module-level streaming thread pool.

    Safe to call multiple times. Call at program exit if you want a clean
    shutdown without waiting for the interpreter's atexit handlers.
    """
    global _STREAM_EXECUTOR
    if _STREAM_EXECUTOR is not None:
        _STREAM_EXECUTOR.shutdown(wait=wait)
        _STREAM_EXECUTOR = None


__all__ = [
    "Engine",
    "GenerateConfig",
    "GbnfConstraint",
    "Metrics",
    "TokenChannel",
    "astream",
    "shutdown_stream_executor",
    "utils",
    "__version__",
]

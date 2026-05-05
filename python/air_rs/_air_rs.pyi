"""Type stubs for air_rs._air_rs (the native Rust extension).

These stubs provide full IDE completion and mypy/pyright checking for
all public classes. The actual implementation is in src/python.rs.
"""

from __future__ import annotations
from typing import AsyncGenerator, Optional

__version__: str

class GenerateConfig:
    """Sampling parameters for a single generation call."""

    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    stop_strings: list[str]
    grammar: GbnfConstraint | None

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stop_strings: list[str] | None = None,
        grammar: GbnfConstraint | None = None,
    ) -> None: ...

class GbnfConstraint:
    """GBNF grammar constraint for structured text generation."""

    @classmethod
    def json_mode(cls) -> GbnfConstraint:
        """Constrain output to valid JSON."""
        ...

    @classmethod
    def integer(cls) -> GbnfConstraint:
        """Constrain output to an integer."""
        ...

    @classmethod
    def identifier(cls) -> GbnfConstraint:
        """Constrain output to a C-style identifier."""
        ...

    @classmethod
    def choice(cls, options: list[str]) -> GbnfConstraint:
        """Constrain output to one of the provided strings."""
        ...

    @classmethod
    def from_grammar(cls, grammar: str) -> GbnfConstraint:
        """Build a constraint from a raw GBNF grammar string."""
        ...

class Metrics:
    """Read-only inference metrics snapshot."""

    tokens_per_second: float
    time_to_first_token_ms: float
    total_time_ms: float
    prompt_tokens: int
    generated_tokens: int

class TokenChannel:
    """Per-request token stream returned by ``Engine._stream_channel()``.

    Not constructed directly — obtain via ``Engine._stream_channel()`` and
    consume through ``air_rs.astream()``.
    """

    def recv_sync(self) -> str | None:
        """Block (GIL released) until the next token is ready.

        Returns the decoded token string, or ``None`` when generation is done.

        Raises
        ------
        RuntimeError
            If the generator hit an internal error.
        """
        ...

    def recv_or_stop(self) -> str:
        """Like ``recv_sync`` but raises ``StopAsyncIteration`` on exhaustion.

        Used internally by ``air_rs.astream()``.
        """
        ...

class Engine:
    """High-performance LLM inference engine."""

    @classmethod
    def from_gguf(
        cls,
        path: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> Engine:
        """Load a model from a GGUF file."""
        ...

    def generate(
        self,
        prompt: str,
        config: GenerateConfig | None = None,
    ) -> str:
        """Generate a complete response string."""
        ...

    def stream_to_list(
        self,
        prompt: str,
        config: GenerateConfig | None = None,
    ) -> list[str]:
        """Stream generated tokens as a list of strings."""
        ...

    def _stream_channel(
        self,
        prompt: str,
        config: GenerateConfig | None = None,
    ) -> TokenChannel:
        """Start generation and return a live ``TokenChannel``.

        Generation runs synchronously in a GIL-free thread; tokens are
        buffered in the channel as they are produced.  Consume via
        ``air_rs.astream()`` for a proper ``async for`` interface.
        """
        ...

    def set_grammar(self, constraint: GbnfConstraint) -> None:
        """Set a persistent grammar constraint for all subsequent calls."""
        ...

    def clear_grammar(self) -> None:
        """Remove the persistent grammar constraint."""
        ...

    def has_grammar(self) -> bool:
        """True if a persistent grammar constraint is active."""
        ...

    def reset(self) -> None:
        """Reset the KV cache (start a fresh conversation)."""
        ...

    def metrics(self) -> Metrics:
        """Return metrics from the last generation call."""
        ...

# ---------------------------------------------------------------------------
# Free async helper — re-exported from air_rs.__init__
# ---------------------------------------------------------------------------

async def astream(
    engine: Engine,
    prompt: str,
    config: GenerateConfig | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding one decoded token per iteration.

    Wraps ``Engine._stream_channel()`` + a thread-pool executor so the
    Python event loop never blocks during generation.

    Examples
    --------
    >>> async for token in air_rs.astream(engine, "Once upon a time"):
    ...     print(token, end="", flush=True)
    """
    ...

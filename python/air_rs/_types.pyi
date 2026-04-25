"""Type stubs for air_rs._air_rs (the native Rust extension).

These stubs provide full IDE completion and mypy/pyright checking for
all public classes. The actual implementation is in src/python.rs.
"""

from __future__ import annotations

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

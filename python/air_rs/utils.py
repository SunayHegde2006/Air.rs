"""Pure-Python utilities for Air.rs.

These helpers are independent of the Rust extension — they work at the
string level and do not require a loaded engine.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    "chatml": "<|im_start|>{role}\n{content}<|im_end|>\n",
    "llama3": "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
    "mistral": "[INST] {content} [/INST]",  # simplified single-turn
    "gemma": "<start_of_turn>{role}\n{content}<end_of_turn>\n",
    "phi3": "<|{role}|>\n{content}<|end|>\n",
}

_BOS: dict[str, str] = {
    "chatml":  "",
    "llama3":  "<|begin_of_text|>",
    "mistral": "<s>",
    "gemma":   "<bos>",
    "phi3":    "",
}

_ASST_START: dict[str, str] = {
    "chatml":  "<|im_start|>assistant\n",
    "llama3":  "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "mistral": "",
    "gemma":   "<start_of_turn>model\n",
    "phi3":    "<|assistant|>\n",
}


def format_chat(
    messages: list[dict[str, str]],
    template: str = "chatml",
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Format a list of chat messages as a prompt string.

    Parameters
    ----------
    messages:
        List of dicts with keys ``"role"`` and ``"content"``.
        Roles are typically ``"system"``, ``"user"``, ``"assistant"``.
    template:
        One of ``"chatml"``, ``"llama3"``, ``"mistral"``, ``"gemma"``, ``"phi3"``.
    add_generation_prompt:
        If True, append the assistant-start token so the model continues
        as the assistant. Disable when constructing non-interactive prompts.

    Returns
    -------
    str
        The formatted prompt string ready to pass to ``Engine.generate()``.

    Examples
    --------
    >>> from air_rs.utils import format_chat
    >>> prompt = format_chat([
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user",   "content": "What is 2+2?"},
    ... ], template="chatml")
    """
    tmpl_key = template.lower()
    if tmpl_key not in _TEMPLATES:
        supported = ", ".join(sorted(_TEMPLATES))
        raise ValueError(f"Unknown template {template!r}. Supported: {supported}")

    tmpl     = _TEMPLATES[tmpl_key]
    bos      = _BOS[tmpl_key]
    asst_tok = _ASST_START[tmpl_key]

    parts: list[str] = [bos] if bos else []

    if tmpl_key == "mistral":
        # Mistral only wraps user turns; system content prepended to first user
        system_prefix = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                system_prefix = content + "\n\n"
            elif role == "user":
                parts.append(tmpl.format(content=system_prefix + content))
                system_prefix = ""
            elif role == "assistant":
                parts.append(f" {content}</s>")
    else:
        for msg in messages:
            parts.append(tmpl.format(role=msg["role"], content=msg["content"]))

    if add_generation_prompt:
        parts.append(asst_tok)

    return "".join(parts)


def count_tokens_approx(text: str) -> int:
    """Rough token count estimate (characters / 4, no tokenizer required).

    Accurate to within ~20% for English text. For precise counts, use the
    engine's tokenizer directly.
    """
    return max(1, len(text) // 4)

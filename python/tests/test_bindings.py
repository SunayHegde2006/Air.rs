"""Tests for air_rs Python bindings."""

from __future__ import annotations

import pytest


class TestUtils:
    def test_chatml_basic(self) -> None:
        from air_rs.utils import format_chat
        r = format_chat([{"role": "user", "content": "Hello"}], template="chatml")
        assert "<|im_start|>user" in r
        assert "Hello" in r
        assert '<|im_start|>assistant' in r

    def test_llama3_bos(self) -> None:
        from air_rs.utils import format_chat
        r = format_chat([{"role": "user", "content": "Hi"}], template="llama3")
        assert "<|begin_of_text|>" in r
        assert "<|start_header_id|>user<|end_header_id|>" in r

    def test_mistral_system_prepended(self) -> None:
        from air_rs.utils import format_chat
        r = format_chat(
            [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hi"}],
            template="mistral",
        )
        assert "Be helpful." in r
        assert "[INST]" in r

    def test_no_generation_prompt(self) -> None:
        from air_rs.utils import format_chat
        r = format_chat(
            [{"role": "user", "content": "Hi"}],
            template="chatml",
            add_generation_prompt=False,
        )
        assert not r.rstrip().endswith("<|im_start|>assistant")

    def test_unknown_template_raises(self) -> None:
        from air_rs.utils import format_chat
        with pytest.raises(ValueError, match="Unknown template"):
            format_chat([], template="nope_xyz")

    def test_phi3_template(self) -> None:
        from air_rs.utils import format_chat
        r = format_chat([{"role": "user", "content": "1+1"}], template="phi3")
        assert "user" in r
        assert "1+1" in r

    def test_gemma_template(self) -> None:
        from air_rs.utils import format_chat
        r = format_chat([{"role": "user", "content": "Test"}], template="gemma")
        assert "<start_of_turn>user" in r

    def test_count_tokens_approx(self) -> None:
        from air_rs.utils import count_tokens_approx
        assert count_tokens_approx("hello world") >= 1
        assert count_tokens_approx("") == 1  # min 1
        # 400 chars -> ~100 tokens
        assert count_tokens_approx("a" * 400) == 100


class TestExtensionSmoke:
    """Smoke tests -- skipped if extension not built."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ext(self) -> None:
        try:
            import air_rs
            if not getattr(air_rs, "_EXTENSION_LOADED", False):
                pytest.skip("Rust extension not compiled")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Rust extension not compiled")


    def test_version_string(self) -> None:
        import air_rs
        assert isinstance(air_rs.__version__, str)
        parts = air_rs.__version__.split(".")
        assert len(parts) >= 2

    def test_generate_config_defaults(self) -> None:
        import air_rs
        cfg = air_rs.GenerateConfig()
        assert cfg.max_tokens == 512
        assert cfg.temperature == pytest.approx(0.7, abs=0.01)
        assert cfg.top_p == pytest.approx(0.9, abs=0.01)
        assert cfg.top_k == 40
        assert cfg.stop_strings == []
        assert cfg.grammar is None

    def test_generate_config_custom(self) -> None:
        import air_rs
        cfg = air_rs.GenerateConfig(max_tokens=128, temperature=0.0)
        assert cfg.max_tokens == 128
        assert cfg.temperature == 0.0

    def test_gbnf_json_mode(self) -> None:
        import air_rs
        c = air_rs.GbnfConstraint.json_mode()
        assert c is not None
        assert "json" in repr(c).lower()

    def test_gbnf_integer(self) -> None:
        import air_rs
        c = air_rs.GbnfConstraint.integer()
        assert c is not None

    def test_gbnf_identifier(self) -> None:
        import air_rs
        c = air_rs.GbnfConstraint.identifier()
        assert c is not None

    def test_gbnf_choice_empty_raises(self) -> None:
        import air_rs
        with pytest.raises(ValueError):
            air_rs.GbnfConstraint.choice([])

    def test_gbnf_choice_valid(self) -> None:
        import air_rs
        c = air_rs.GbnfConstraint.choice(["yes", "no"])
        assert "yes" in repr(c)

    def test_gbnf_from_grammar_invalid_raises(self) -> None:
        import air_rs
        with pytest.raises(ValueError, match="GBNF"):
            air_rs.GbnfConstraint.from_grammar("this is not valid gbnf")

    def test_gbnf_from_grammar_valid(self) -> None:
        import air_rs
        c = air_rs.GbnfConstraint.from_grammar('root ::= [a-z]+')
        assert c is not None


class TestEngineIntegration:
    """Integration tests -- require a real GGUF file."""

    @pytest.fixture(autouse=True)
    def skip_without_model(self, tmp_path) -> None:
        import os
        model = os.environ.get("AIR_RS_TEST_MODEL")
        if not model or not os.path.exists(model):
            pytest.skip("Set AIR_RS_TEST_MODEL=path/to/model.gguf to run integration tests")
        self.model_path = model

    def test_generate_returns_string(self) -> None:
        import air_rs
        engine = air_rs.Engine.from_gguf(self.model_path)
        result = engine.generate("Hello", config=air_rs.GenerateConfig(max_tokens=16))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_reset_clears_kv_cache(self) -> None:
        import air_rs
        engine = air_rs.Engine.from_gguf(self.model_path)
        engine.generate("A", config=air_rs.GenerateConfig(max_tokens=4))
        engine.reset()  # should not raise

    def test_metrics_after_generate(self) -> None:
        import air_rs
        engine = air_rs.Engine.from_gguf(self.model_path)
        engine.generate("2+2=", config=air_rs.GenerateConfig(max_tokens=4))
        m = engine.metrics()
        assert m.tokens_per_second > 0
        assert m.generated_tokens >= 1


# ---------------------------------------------------------------------------
# astream — pure-Python unit tests (no model required)
# ---------------------------------------------------------------------------

class TestAstreamUnit:
    """Tests for the astream() async generator helper — no GGUF required."""

    def test_astream_is_async_generator_function(self) -> None:
        import inspect

        import air_rs
        assert inspect.isasyncgenfunction(air_rs.astream)

    def test_astream_in_all(self) -> None:
        import air_rs
        assert "astream" in air_rs.__all__

    def test_token_channel_in_all(self) -> None:
        import air_rs
        assert "TokenChannel" in air_rs.__all__

    def test_shutdown_stream_executor_callable(self) -> None:
        import air_rs
        assert callable(air_rs.shutdown_stream_executor)

    def test_shutdown_stream_executor_idempotent(self) -> None:
        import air_rs
        # First call tears down; second should be a no-op
        air_rs.shutdown_stream_executor(wait=False)
        air_rs.shutdown_stream_executor(wait=False)  # must not raise

    def test_get_executor_creates_thread_pool(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from air_rs import _get_executor
        pool = _get_executor()
        assert isinstance(pool, ThreadPoolExecutor)
        # Second call returns same instance
        assert _get_executor() is pool

    def test_astream_no_ext_raises_import_error(self, monkeypatch) -> None:
        """When extension not loaded, astream() raises ImportError."""
        import asyncio

        import air_rs

        monkeypatch.setattr(air_rs, "_EXTENSION_LOADED", False)

        async def run():
            # astream() is an async generator — calling it returns the gen object,
            # first __anext__ triggers the body and should raise ImportError.
            gen = air_rs.astream(object(), "hello")  # type: ignore
            with pytest.raises(ImportError, match="Rust extension"):
                await gen.__anext__()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# astream — integration tests (model-gated)
# ---------------------------------------------------------------------------

class TestAstreamIntegration:
    """Integration tests for astream() — require a real GGUF model."""

    @pytest.fixture(autouse=True)
    def skip_without_model(self, tmp_path) -> None:
        import os
        model = os.environ.get("AIR_RS_TEST_MODEL")
        if not model or not os.path.exists(model):
            pytest.skip("Set AIR_RS_TEST_MODEL=path/to/model.gguf to run async integration tests")
        self.model_path = model

    def test_astream_yields_strings(self) -> None:
        import asyncio

        import air_rs

        async def run():
            engine = air_rs.Engine.from_gguf(self.model_path)
            tokens = []
            cfg = air_rs.GenerateConfig(max_tokens=16)
            async for token in air_rs.astream(engine, "Once upon a time", cfg):
                assert isinstance(token, str)
                tokens.append(token)
            assert len(tokens) >= 1

        asyncio.run(run())

    def test_astream_concatenates_to_nonempty(self) -> None:
        import asyncio

        import air_rs

        async def run():
            engine = air_rs.Engine.from_gguf(self.model_path)
            cfg = air_rs.GenerateConfig(max_tokens=8)
            result = ""
            async for token in air_rs.astream(engine, "The sky is", cfg):
                result += token
            assert len(result) > 0

        asyncio.run(run())

    def test_astream_event_loop_stays_alive(self) -> None:
        """Other coroutines must run while astream is active."""
        import asyncio

        import air_rs

        ran_other = []

        async def other_coro():
            ran_other.append(1)

        async def run():
            engine = air_rs.Engine.from_gguf(self.model_path)
            cfg = air_rs.GenerateConfig(max_tokens=8)
            # Schedule the other coroutine; it should run between token yields
            task = asyncio.create_task(other_coro())
            async for _ in air_rs.astream(engine, "Hello", cfg):
                await asyncio.sleep(0)  # yield to allow other_coro to run
            await task

        asyncio.run(run())
        assert ran_other, "event loop did not run other tasks during astream"


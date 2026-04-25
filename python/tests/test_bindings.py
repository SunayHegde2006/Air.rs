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
        r = format_chat([{"role": "user", "content": "Hi"}], template="chatml", add_generation_prompt=False)
        assert not r.rstrip().endswith('<|im_start|>assistant')

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

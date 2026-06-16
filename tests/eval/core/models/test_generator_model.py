import os
import unittest
from unittest.mock import MagicMock, patch

from flintai.eval.core.models.model_retry import RetryModel


class TestGetGeneratorModel(unittest.TestCase):

    def test_missing_env_var_raises(self):
        from flintai.eval.core.models.generator_model import get_generator_model
        env = os.environ.copy()
        env.pop("GENERATOR_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                get_generator_model()
            self.assertIn("GENERATOR_MODEL", str(ctx.exception))

    @patch.dict(os.environ, {"GENERATOR_MODEL": "nocolon"})
    def test_invalid_format_no_colon_raises(self):
        from flintai.eval.core.models.generator_model import get_generator_model
        with self.assertRaises(ValueError) as ctx:
            get_generator_model()
        self.assertIn("Invalid GENERATOR_MODEL format", str(ctx.exception))

    @patch.dict(os.environ, {"GENERATOR_MODEL": "unknown_type:model"})
    def test_unknown_type_raises(self):
        from flintai.eval.core.models.generator_model import get_generator_model
        with self.assertRaises(ValueError) as ctx:
            get_generator_model()
        self.assertIn("Unknown generator model type", str(ctx.exception))

    @patch.dict(os.environ, {"GENERATOR_MODEL": "openai:gpt-4o"})
    @patch("flintai.eval.core.models.generator_model._create_inner")
    def test_returns_retry_model(self, mock_create_inner: MagicMock):
        mock_create_inner.return_value = MagicMock()
        from flintai.eval.core.models.generator_model import get_generator_model
        result = get_generator_model()
        self.assertIsInstance(result, RetryModel)

    @patch.dict(os.environ, {"GENERATOR_MODEL": "openai:gpt-4o"})
    @patch("flintai.eval.core.models.generator_model._create_inner")
    def test_passes_model_type_and_name(self, mock_create_inner: MagicMock):
        mock_create_inner.return_value = MagicMock()
        from flintai.eval.core.models.generator_model import get_generator_model
        get_generator_model()
        mock_create_inner.assert_called_once_with("openai", "gpt-4o")

    @patch.dict(os.environ, {"GENERATOR_MODEL": "openai: gpt-4o "})
    @patch("flintai.eval.core.models.generator_model._create_inner")
    def test_strips_whitespace(self, mock_create_inner: MagicMock):
        mock_create_inner.return_value = MagicMock()
        from flintai.eval.core.models.generator_model import get_generator_model
        get_generator_model()
        mock_create_inner.assert_called_once_with("openai", "gpt-4o")


class TestCreateInner(unittest.TestCase):

    def test_gemini(self):
        from flintai.eval.core.models.generator_model import _create_inner
        with patch("flintai.eval.core.models.model_gemini.GeminiModel") as mock_model:
            with patch("google.genai.Client") as mock_client:
                result = _create_inner("gemini", "gemini-2.5-flash-lite")
                mock_client.assert_called_once()
                mock_model.assert_called_once()
                self.assertIsNotNone(result)

    def test_openai(self):
        from flintai.eval.core.models.generator_model import _create_inner
        with patch("flintai.eval.core.models.model_openai.OpenAIModel") as mock_model:
            with patch("openai.AsyncOpenAI") as mock_client:
                result = _create_inner("openai", "gpt-4o")
                mock_client.assert_called_once()
                mock_model.assert_called_once()
                self.assertIsNotNone(result)

    def test_anthropic(self):
        from flintai.eval.core.models.generator_model import _create_inner
        with patch("flintai.eval.core.models.model_anthropic.AnthropicModel") as mock_model:
            with patch("anthropic.AsyncAnthropic") as mock_client:
                result = _create_inner("anthropic", "claude-haiku-4-20250414")
                mock_client.assert_called_once()
                mock_model.assert_called_once()
                self.assertIsNotNone(result)

    def test_litellm(self):
        from flintai.eval.core.models.generator_model import _create_inner
        with patch("flintai.eval.core.models.model_litellm.LiteLLMModel") as mock_model:
            result = _create_inner("litellm", "gpt-4o")
            mock_model.assert_called_once_with("gpt-4o")
            self.assertIsNotNone(result)

    def test_ollama(self):
        from flintai.eval.core.models.generator_model import _create_inner
        with patch("flintai.eval.core.models.model_ollama.OllamaModel") as mock_model:
            result = _create_inner("ollama", "llama3")
            mock_model.assert_called_once_with("llama3")
            self.assertIsNotNone(result)

    def test_unknown_type(self):
        from flintai.eval.core.models.generator_model import _create_inner
        with self.assertRaises(ValueError) as ctx:
            _create_inner("foobar", "model")
        self.assertIn("Unknown generator model type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

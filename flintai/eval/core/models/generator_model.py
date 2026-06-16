"""
Global generator/judge model used for LLM-as-judge detection,
metric evaluations, adversarial probing, and evaluation
generation.

Configured via the GENERATOR_MODEL environment variable:
    GENERATOR_MODEL=type:model_name

Examples:
    GENERATOR_MODEL=gemini:gemini-2.5-flash-lite
    GENERATOR_MODEL=openai:gpt-4o-mini
    GENERATOR_MODEL=anthropic:claude-haiku-4-20250414
    GENERATOR_MODEL=litellm:gpt-4o
    GENERATOR_MODEL=ollama:llama3
"""

from __future__ import annotations

import logging
import os

from flintai.eval.core.models.model import Model
from flintai.eval.core.models.model_retry import RetryModel

logger = logging.getLogger(__name__)


def get_generator_model() -> Model:
    """Create a fresh generator model instance.

    Always creates a new instance because async SDK clients
    are bound to the event loop they're first used on.
    Caching across loops causes 'Future attached to a
    different loop' errors.
    """
    return _create_generator_model()


def _create_generator_model() -> Model:
    spec = os.environ.get("GENERATOR_MODEL")
    if not spec:
        raise ValueError(
            "GENERATOR_MODEL environment variable is not set. "
            "Expected format: type:model_name "
            "(e.g. gemini:gemini-2.5-flash-lite)"
        )

    if ":" not in spec:
        raise ValueError(
            f"Invalid GENERATOR_MODEL format: {spec!r}. "
            "Expected type:model_name"
        )

    model_type, model_name = spec.split(":", 1)
    inner = _create_inner(model_type.strip(), model_name.strip())
    logger.debug("Creating generator model: %s", spec)
    return RetryModel(inner, max_retries=5)


def _create_inner(model_type: str, model_name: str) -> Model:
    if model_type == "gemini":
        from google.genai import Client
        from flintai.eval.core.models.model_gemini import GeminiModel
        return GeminiModel(Client(), model_name)

    elif model_type == "openai":
        from openai import AsyncOpenAI
        from flintai.eval.core.models.model_openai import OpenAIModel
        return OpenAIModel(AsyncOpenAI(), model_name)

    elif model_type == "anthropic":
        from anthropic import AsyncAnthropic
        from flintai.eval.core.models.model_anthropic import AnthropicModel
        return AnthropicModel(AsyncAnthropic(), model_name)

    elif model_type == "litellm":
        from flintai.eval.core.models.model_litellm import LiteLLMModel
        return LiteLLMModel(model_name)

    elif model_type == "ollama":
        from flintai.eval.core.models.model_ollama import OllamaModel
        return OllamaModel(model_name)

    else:
        raise ValueError(
            f"Unknown generator model type: {model_type!r}. "
            "Supported: gemini, openai, anthropic, litellm, ollama"
        )

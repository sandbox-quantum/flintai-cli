from flintai.eval.common.utils import resolve_env, resolve_env_dict
from flintai.eval.core.models.model import Model
from flintai.eval.core.models.model_retry import RetryModel
from flintai.eval.db.base.models.model_types import DbModel, ModelType


def create_model(db_model: DbModel, max_retries: int = 5) -> Model:
    """Create a Model instance from a DbModel,
    wrapped in a RetryModel for transient error handling."""
    return RetryModel(
        _create_inner_model(db_model),
        max_retries=max_retries,
    )


def _create_inner_model(db_model: DbModel) -> Model:
    temp = db_model.temperature
    key = resolve_env(db_model.key)
    headers = resolve_env_dict(db_model.headers)

    if db_model.type == ModelType.ANTHROPIC:
        from anthropic import AsyncAnthropic

        from flintai.eval.core.models.model_anthropic import (
            AnthropicModel,
        )

        client_kwargs = {}
        if key:
            client_kwargs["api_key"] = key

        return AnthropicModel(
            AsyncAnthropic(**client_kwargs),
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.OPENAI:
        from openai import AsyncOpenAI

        from flintai.eval.core.models.model_openai import OpenAIModel

        client_kwargs = {}
        if key:
            client_kwargs["api_key"] = key

        return OpenAIModel(
            AsyncOpenAI(**client_kwargs),
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.GEMINI:
        from google.genai import Client

        from flintai.eval.core.models.model_gemini import GeminiModel

        client_kwargs = {}
        if key:
            client_kwargs["api_key"] = key

        return GeminiModel(
            Client(**client_kwargs),
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.LITELLM:
        from flintai.eval.core.models.model_litellm import (
            LiteLLMModel,
        )

        return LiteLLMModel(
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.HUGGINGFACE:
        from flintai.eval.core.models.model_huggingface import (
            HuggingFaceModel,
        )

        return HuggingFaceModel(
            db_model.model_name,
            token=key or None,
            temperature=temp,
        )

    elif db_model.type == ModelType.OLLAMA:
        from flintai.eval.core.models.model_ollama import (
            OllamaModel,
        )

        return OllamaModel(
            db_model.model_name,
            host=db_model.host or "http://localhost:11434",
            temperature=temp,
        )

    elif db_model.type == ModelType.ADK:
        from flintai.eval.core.models.model_adk import ADKModel

        return ADKModel(
            app_name=db_model.model_name,
            host=db_model.host or "http://localhost:8000",
            immediate_result=db_model.immediate_result,
        )

    elif db_model.type == ModelType.OPENAI_AGENT:
        from flintai.eval.core.models.model_openai_agent import (
            OpenAIAgentModel,
        )

        return OpenAIAgentModel(
            host=db_model.host or "http://localhost:8000",
            endpoint=db_model.endpoint or "/run",
        )

    elif db_model.type == ModelType.ANTHROPIC_AGENT:
        from flintai.eval.core.models.model_anthropic_agent import (
            AnthropicAgentModel,
        )

        return AnthropicAgentModel(
            host=db_model.host or "http://localhost:8000",
            endpoint=db_model.endpoint or "/run",
        )

    elif db_model.type == ModelType.OPENAI_COMPATIBLE:
        from flintai.eval.core.models.model_openai_compatible import (
            OpenAICompatibleModel,
        )

        return OpenAICompatibleModel(
            base_url=db_model.host or "http://localhost:8000/v1",
            model=db_model.model_name,
            api_key=key or "EMPTY",
            temperature=temp,
            headers=headers or None,
        )

    elif db_model.type == ModelType.GENERIC_HTTP:
        from flintai.eval.core.models.model_generic_http import (
            GenericHttpModel,
        )

        url = db_model.host or "http://localhost:8000"
        if db_model.endpoint:
            url = f"{url.rstrip('/')}{db_model.endpoint}"

        return GenericHttpModel(
            url=url,
            headers=headers or None,
            input_path=db_model.input_path or "input",
            output_path=db_model.output_path or "output",
        )

    elif db_model.type == ModelType.LANGSERVE:
        from flintai.eval.core.models.model_langserve import (
            LangServeModel,
        )

        return LangServeModel(
            base_url=db_model.host or "http://localhost:8000",
            chain_path=db_model.endpoint or "",
            headers=headers or None,
        )

    else:
        raise ValueError(
            f"unknown model type: {db_model.type}"
        )

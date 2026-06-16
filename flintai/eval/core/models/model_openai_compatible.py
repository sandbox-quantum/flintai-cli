"""Model for any OpenAI-compatible chat completions endpoint.

Works with vLLM, SGLang, Ollama, LiteLLM proxy, Azure
OpenAI, Amazon Bedrock, TGI, and any other server that
implements the ``/v1/chat/completions`` API.
"""

from typing import Any

from openai import AsyncOpenAI

from flintai.eval.common import converter_openai
from flintai.eval.common.schema import Message
from flintai.eval.core.models.model import Model, ModelResponse


class OpenAICompatibleModel(Model):
    _client: AsyncOpenAI
    _model: str
    _temperature: float

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        headers: dict[str, str] | None = None,
    ):
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            default_headers=headers or {},
        )
        self._model = model
        self._temperature = temperature

    async def _generate(
        self, messages: list[Message], **kwargs: Any,
    ) -> ModelResponse:
        openai_messages = [
            converter_openai.from_message(m)
            for m in messages
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            temperature=kwargs.pop(
                "temperature", self._temperature,
            ),
            **kwargs,
        )
        choice = response.choices[0]
        message = converter_openai.to_message(
            choice.message,
        )
        return ModelResponse(message)

from typing import Any

import requests
from openai import AsyncOpenAI

from flintai.eval.common import converter_openai
from flintai.eval.common.schema import Message
from flintai.eval.core.models.model import Model, ModelResponse


class OllamaModel(Model):
    _client: AsyncOpenAI
    _model: str
    _temperature: float

    def __init__(
        self, model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self._client = AsyncOpenAI(
            base_url=f"{host}/v1",
            api_key="ollama",
        )
        self._model = model
        self._temperature = temperature

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        openai_messages = [
            converter_openai.from_message(m) for m in messages
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            temperature=kwargs.pop("temperature", self._temperature),
            **kwargs,
        )
        choice = response.choices[0]
        message = converter_openai.to_message(choice.message)
        return ModelResponse(message)


def discover_ollama_models(
    host: str = "http://localhost:11434",
) -> list[str]:
    """Return a list of model names available on an Ollama instance."""
    response = requests.get(f"{host}/api/tags")
    response.raise_for_status()
    data = response.json()
    return [m["name"] for m in data.get("models", [])]

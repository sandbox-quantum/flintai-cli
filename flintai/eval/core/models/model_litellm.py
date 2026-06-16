from typing import Any

import litellm

from flintai.eval.common import converter_openai
from flintai.eval.common.schema import Message
from flintai.eval.core.models.model import Model, ModelResponse


class LiteLLMModel(Model):
    """Model adapter that delegates to LiteLLM.

    LiteLLM uses the OpenAI message format for all providers, so this
    adapter reuses ``converter_openai`` for serialisation.
    """

    _model: str
    _temperature: float

    def __init__(
        self, model: str, temperature: float = 0.0,
    ):
        self._model = model
        self._temperature = temperature

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        openai_messages = [
            converter_openai.from_message(m) for m in messages
        ]
        response = await litellm.acompletion(
            model=self._model,
            messages=openai_messages,
            temperature=kwargs.pop("temperature", self._temperature),
            **kwargs,
        )
        choice = response.choices[0]
        message = converter_openai.to_message(choice.message)
        return ModelResponse(message)

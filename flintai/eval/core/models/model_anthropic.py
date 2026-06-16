from typing import Any

from anthropic import AsyncAnthropic

from flintai.eval.common import converter_anthropic
from flintai.eval.common.schema import Message, PartType, Role
from flintai.eval.core.models.model import Model, ModelResponse


class AnthropicModel(Model):
    _client: AsyncAnthropic
    _model: str
    _max_tokens: int
    _temperature: float

    def __init__(
        self, client: AsyncAnthropic, model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        system_blocks = None
        api_messages = []
        for msg in messages:
            anthropic_msg = converter_anthropic.from_message(msg)
            if msg.content.role == Role.SYSTEM:
                system_blocks = anthropic_msg["content"]
            else:
                api_messages.append(anthropic_msg)

        if not api_messages:
            api_messages = [{"role": "user", "content": ""}]

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "temperature": kwargs.pop("temperature", self._temperature),
            "messages": api_messages,
            **kwargs,
        }
        if system_blocks is not None:
            create_kwargs["system"] = system_blocks

        response = await self._client.messages.create(**create_kwargs)
        message = converter_anthropic.to_message(response)
        return ModelResponse(message)

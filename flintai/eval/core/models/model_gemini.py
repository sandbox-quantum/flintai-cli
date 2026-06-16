from typing import Any

from google.genai import Client, types as genai_types

from flintai.eval.common import converter_genai
from flintai.eval.common.schema import Message, Role
from flintai.eval.core.models.model import Model, ModelResponse, ResponseStatus


class GeminiModel(Model):
    _client: Client
    _model: str
    _temperature: float

    def __init__(
        self, client: Client, model: str,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model = model
        self._temperature = temperature

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.content.role == Role.SYSTEM:
                system_parts.extend(
                    converter_genai.from_content(
                        msg.content,
                    )["parts"]
                )
            else:
                contents.append(
                    converter_genai.from_content(msg.content),
                )

        config = kwargs.pop("config", None)
        if config is None:
            config = genai_types.GenerateContentConfig(
                temperature=self._temperature,
            )
        elif config.temperature is None:
            config.temperature = self._temperature
        if system_parts:
            system_instruction = genai_types.Content(
                parts=[
                    genai_types.Part(**p)
                    for p in system_parts
                ],
            )
            config.system_instruction = system_instruction

        generate_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": contents,
            **kwargs,
        }
        if config is not None:
            generate_kwargs["config"] = config

        response = await self._client.aio.models.generate_content(
            **generate_kwargs,
        )
        if (
            response.candidates is not None
            and len(response.candidates) == 1
            and response.candidates[0].content is not None
        ):
            message = converter_genai.to_message(
                response.candidates[0].content,
            )
            return ModelResponse(message)
        return ModelResponse(
            None, _classify_block_reason(response),
        )


def _classify_block_reason(response) -> ResponseStatus:
    """Determine why a Gemini response was blocked."""
    # Check prompt-level blocking
    reason = getattr(
        response, "prompt_feedback", None,
    )
    if reason is not None:
        block_reason = getattr(reason, "block_reason", None)
        if block_reason is not None:
            reason_str = str(block_reason).upper()
            if "SAFETY" in reason_str:
                return ResponseStatus.BLOCKED_SAFETY
            if "PROHIBITED" in reason_str:
                return ResponseStatus.BLOCKED_PROHIBITED

    # Check candidate-level finish reason
    candidates = getattr(response, "candidates", None)
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        if finish is not None:
            finish_str = str(finish).upper()
            if "SAFETY" in finish_str:
                return ResponseStatus.BLOCKED_SAFETY
            if "RECITATION" in finish_str:
                return ResponseStatus.BLOCKED_RECITATION
            if "PROHIBITED" in finish_str:
                return ResponseStatus.BLOCKED_PROHIBITED

    return ResponseStatus.BLOCKED_SAFETY

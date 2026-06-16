import asyncio
from typing import Any

from transformers import Pipeline, pipeline

from flintai.eval.common.schema import Content, Message, PartType, Role
from flintai.eval.core.models.model import Model, ModelResponse


class HuggingFaceModel(Model):
    """Model adapter for Hugging Face ``text-generation`` pipelines.

    Accepts either an already-constructed ``Pipeline`` or a model name /
    path to load one on the fly.
    """

    _pipeline: Pipeline
    _temperature: float

    def __init__(
        self, model: str | Pipeline,
        temperature: float = 0.0,
        **pipeline_kwargs: Any,
    ):
        if isinstance(model, str):
            self._pipeline = pipeline(
                "text-generation", model=model,
                **pipeline_kwargs,
            )
        else:
            self._pipeline = model
        self._temperature = temperature

    async def _generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return await asyncio.to_thread(
            self._generate_sync, messages, **kwargs,
        )

    def _generate_sync(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        if len(messages) > 1:
            raise ValueError(
                "HuggingFaceModel does not support "
                "multiple messages"
            )
        prompt = _collect_text(messages[0].content)
        kwargs.setdefault("max_new_tokens", 256)
        kwargs.setdefault("return_full_text", False)
        kwargs.setdefault("temperature", self._temperature or None)
        outputs = self._pipeline(prompt, **kwargs)
        generated_text = outputs[0]["generated_text"]
        message = Message(
            content=Content.text(Role.ASSISTANT, generated_text),
        )
        return ModelResponse(message)


def _collect_text(content: Content) -> str:
    return " ".join(
        p.text for p in content.parts
        if p.part_type == PartType.TEXT and p.text
    )

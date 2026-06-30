import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from flintai.eval.common.schema import PartType
from flintai.eval.common.schema import Content, Message, Role

logger = logging.getLogger(__name__)


class ResponseStatus(str, Enum):
    """Outcome of a model generation request."""

    OK = "ok"
    BLOCKED_SAFETY = "blocked_safety"
    BLOCKED_RECITATION = "blocked_recitation"
    BLOCKED_PROHIBITED = "blocked_prohibited"
    EMPTY_RESPONSE = "empty_response"
    ERROR = "error"


type ModelContent = str | Message | list[Message]


class ModelResponse:
    message: Message | None
    status: ResponseStatus

    def __init__(
        self,
        message: Message | None,
        status: ResponseStatus = ResponseStatus.OK,
    ):
        self.message = message
        self.status = status


def _message_text(msg: Message) -> str:
    parts = [p.text for p in msg.content.parts if p.text]
    return " ".join(parts) if parts else ""


def _prompt_length(contents: ModelContent) -> int:
    if isinstance(contents, str):
        return len(contents)
    if isinstance(contents, Message):
        return len(_message_text(contents))
    return sum(len(_message_text(m)) for m in contents) if contents else 0


class Model(ABC):
    async def generate(
        self,
        contents: ModelContent,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            contents: The input to the model. Can be:
                - A string (converted to a user message)
                - A single Message
                - A list of Messages
        """
        if isinstance(contents, str):
            messages = [
                Message(
                    content=Content.text(Role.USER, contents),
                ),
            ]
        elif isinstance(contents, Message):
            messages = [contents]
        else:
            messages = contents

        prompt_len = _prompt_length(contents)
        model_name = type(self).__name__

        try:
            response = await self._generate(messages, **kwargs)
        except Exception as e:
            logger.error(
                "%s: prompt=%d chars (%s: %s)",
                model_name, prompt_len, type(e).__name__, e,
            )
            raise

        response_len = (
            len(_message_text(response.message))
            if response.message else 0
        )
        if response.status != ResponseStatus.OK:
            logger.warning(
                "%s: prompt=%d chars, status=%s",
                model_name, prompt_len, response.status.value,
            )
        else:
            logger.debug(
                "%s: prompt=%d chars, response=%d chars",
                model_name, prompt_len, response_len,
            )

        return response

    @abstractmethod
    async def _generate(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> ModelResponse:
        pass


# -- Helper functions -----------------------------------------------


def extract_text_from_message(message: Message) -> str:
    return "".join(
        part.text for part in message.content.parts
        if part.text is not None
    )


def extract_final_text(message: Message) -> str:
    """Extract only the final text output, excluding
    thinking, tool calls, and tool results."""
    text_parts = [
        part.text for part in message.content.parts
        if part.text is not None
        and part.part_type == PartType.TEXT
    ]
    if text_parts:
        return "".join(text_parts)
    return extract_text(message)


def extract_text_from_conversation(
    messages: list[Message],
) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.content.role.value.upper()
        text = extract_final_text(msg)
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)

"""Model for any HTTP endpoint with configurable JSON fields.

Connects to a generic REST API where the request sends
a prompt in a configurable JSON field and the response
returns the output in another configurable field.

Default format::

    POST /your-endpoint
    {"input": "Hello"}

    {"output": "Hi there!"}

The ``output_path`` supports dot-separated paths for
nested responses, e.g. ``"data.response.text"`` or
``"choices.0.message.content"``.
"""

from typing import Any

import aiohttp

from flintai.eval.common.schema import Content, Message, Part, Role
from flintai.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
)


class GenericHttpModel(Model):
    _url: str
    _headers: dict[str, str]
    _input_path: str
    _output_path: str

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        input_path: str = "input",
        output_path: str = "output",
    ):
        self._url = url
        self._headers = headers or {}
        self._input_path = input_path
        self._output_path = output_path

    async def _generate(
        self, messages: list[Message], **kwargs: Any,
    ) -> ModelResponse:
        prompt_text = _flatten_messages(messages)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                json={self._input_path: prompt_text},
                headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        output = _resolve_path(data, self._output_path)
        if not output:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(str(output))],
            ),
        )
        return ModelResponse(message=message)


def _flatten_messages(messages: list[Message]) -> str:
    if len(messages) == 1:
        parts = [
            p.text for p in messages[0].content.parts
            if p.text
        ]
        return " ".join(parts)

    lines: list[str] = []
    for msg in messages:
        role = msg.content.role.value.upper()
        text = " ".join(
            p.text for p in msg.content.parts if p.text
        )
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def _resolve_path(data: Any, path: str) -> Any:
    """Traverse a nested dict/list by dot-separated path.

    Supports integer indices for lists, e.g.
    ``"choices.0.message.content"`` resolves
    ``data["choices"][0]["message"]["content"]``.
    """
    current = data
    for key in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                current = current[int(key)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current

"""Model for LangServe endpoints.

LangServe exposes LangChain runnables as REST APIs with
a standard ``/{chain}/invoke`` endpoint.

Request format::

    POST /my-chain/invoke
    {"input": "Hello"}

Response format::

    {"output": "Hi there!", "metadata": {...}}
"""

from typing import Any

import aiohttp

from flintai.eval.common.schema import Content, Message, Part, Role
from flintai.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
)


class LangServeModel(Model):
    _url: str
    _headers: dict[str, str]

    def __init__(
        self,
        base_url: str,
        chain_path: str = "",
        headers: dict[str, str] | None = None,
    ):
        base = base_url.rstrip("/")
        path = chain_path.strip("/")
        if path:
            self._url = f"{base}/{path}/invoke"
        else:
            self._url = f"{base}/invoke"
        self._headers = headers or {}

    async def _generate(
        self, messages: list[Message], **kwargs: Any,
    ) -> ModelResponse:
        prompt_text = _flatten_messages(messages)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                json={"input": prompt_text},
                headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        output = data.get("output")
        if not output:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        if isinstance(output, dict):
            output = output.get("content", str(output))

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

"""Model implementation for Anthropic Agent SDK agents served over HTTP.

Anthropic Agent SDK (claude_agent_sdk) agents don't have a
built-in HTTP server.  The expected pattern is a FastAPI wrapper
with a POST endpoint that accepts ``{"prompt": "<prompt>"}`` and
returns ``{"response": "<response>"}``.

Example server::

    from fastapi import FastAPI
    from pydantic import BaseModel
    from claude_agent_sdk import (
        query, ClaudeAgentOptions,
        AssistantMessage, TextBlock, ResultMessage,
    )

    app = FastAPI()

    class Request(BaseModel):
        prompt: str

    @app.post("/run")
    async def run(req: Request):
        options = ClaudeAgentOptions(
            permission_mode="acceptEdits",
        )
        text = ""
        async for msg in query(
            prompt=req.prompt, options=options,
        ):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
        return {"response": text}
"""

from typing import Any

import aiohttp

from flintai.eval.common.schema import Content, Message, Part, Role
from flintai.eval.core.models.model import Model, ModelResponse, ResponseStatus


class AnthropicAgentModel(Model):
    """Sends prompts to an Anthropic Agent SDK agent over HTTP."""

    def __init__(
        self,
        host: str = "http://localhost:8000",
        endpoint: str = "/run",
    ):
        self._host = host.rstrip("/")
        self._endpoint = endpoint

    async def _generate(
        self, messages: list[Message], **kwargs: Any,
    ) -> ModelResponse:
        if len(messages) > 1:
            raise ValueError(
                "AnthropicAgentModel does not support "
                "multiple messages"
            )
        text_parts = [
            p.text for p in messages[0].content.parts
            if p.text
        ]
        prompt_text = " ".join(text_parts)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._host}{self._endpoint}",
                json={"prompt": prompt_text},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        output = data.get("response")
        if not output:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(output)],
            ),
        )
        return ModelResponse(message=message)

"""Model implementation for OpenAI Agents SDK agents served over HTTP.

OpenAI Agents SDK agents don't have a built-in HTTP server.
The expected pattern is a FastAPI wrapper with a POST endpoint
that accepts ``{"input": "<prompt>"}`` and returns
``{"output": "<response>"}``.

Example server::

    from fastapi import FastAPI
    from pydantic import BaseModel
    from agents import Agent, Runner

    app = FastAPI()
    agent = Agent(name="my_agent", instructions="...")

    class Request(BaseModel):
        input: str

    @app.post("/run")
    async def run(req: Request):
        result = await Runner.run(agent, req.input)
        return {"output": result.final_output}
"""

from typing import Any

import aiohttp

from flintai.eval.common.schema import Content, Message, Part, Role
from flintai.eval.core.models.model import Model, ModelResponse, ResponseStatus


class OpenAIAgentModel(Model):
    """Sends prompts to an OpenAI Agents SDK agent over HTTP."""

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
                "OpenAIAgentModel does not support "
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
                json={"input": prompt_text},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        output = data.get("output")
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

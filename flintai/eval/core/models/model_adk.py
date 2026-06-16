"""Model implementation for Google ADK agents.

ADK agents are served via ``adk api_server`` or ``adk web`` and
expose a REST API at a configurable host/port.  This model sends
prompts to the ``/run`` endpoint and extracts the agent's text
response from the returned events.

Typical ADK server URL: ``http://localhost:8000``
"""

from typing import Any

import aiohttp
import requests

from flintai.eval.common.schema import Content, Message, Part, Role
from flintai.eval.core.models.model import Model, ModelResponse, ResponseStatus


class ADKModel(Model):
    """Sends prompts to a Google ADK agent served over HTTP.

    Args:
        app_name: The ADK app/agent name.
        host: The HTTP host where the ADK server is running.
        user_id: User identifier for sessions.
        immediate_result: If True, return the first event's
            content (e.g. a tool call). If False (default),
            return the final text response after all tool
            calls complete.
    """

    def __init__(
        self,
        app_name: str,
        host: str = "http://localhost:8000",
        user_id: str = "aired",
        immediate_result: bool = False,
    ):
        self._app_name = app_name
        self._host = host.rstrip("/")
        self._user_id = user_id
        self._immediate_result = immediate_result

    async def _create_session(
        self, session: aiohttp.ClientSession,
    ) -> str:
        """Create a new session for this request."""
        url = (
            f"{self._host}/apps/{self._app_name}"
            f"/users/{self._user_id}/sessions"
        )
        async with session.post(url, json={}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["id"]

    async def _generate(
        self, messages: list[Message], **_kwargs: Any,
    ) -> ModelResponse:
        if len(messages) > 1:
            raise ValueError(
                "ADKModel does not support multiple "
                "messages; use session-based history"
            )

        async with aiohttp.ClientSession() as session:
            session_id = await self._create_session(session)

            text_parts = [
                p.text for p in messages[0].content.parts
                if p.text
            ]
            prompt_text = " ".join(text_parts)

            payload = {
                "appName": self._app_name,
                "userId": self._user_id,
                "sessionId": session_id,
                "newMessage": {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                },
            }

            async with session.post(
                f"{self._host}/run",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                events = await resp.json()

        if self._immediate_result:
            response_text = self._extract_first(events)
        else:
            response_text = self._extract_final(events)

        if response_text is None:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(response_text)],
            ),
        )
        return ModelResponse(message=message)

    def _extract_final(
        self, events: list[dict],
    ) -> str | None:
        """Extract the last text response, skipping tool
        call/response events."""
        for event in reversed(events):
            content = event.get("content")
            if content is None:
                continue
            parts = content.get("parts", [])
            # Skip events that are tool calls or responses
            if any("functionCall" in p or "functionResponse" in p
                   for p in parts if isinstance(p, dict)):
                continue
            texts = [
                p["text"] for p in parts
                if isinstance(p, dict) and "text" in p
            ]
            if texts:
                return " ".join(texts)
        return None

    def _extract_first(
        self, events: list[dict],
    ) -> str | None:
        """Extract the first event's text content."""
        for event in events:
            content = event.get("content")
            if content is None:
                continue
            parts = content.get("parts", [])
            texts = [
                p["text"] for p in parts
                if isinstance(p, dict) and "text" in p
            ]
            if texts:
                return " ".join(texts)
        return None


def discover_adk_agents(
    host: str = "http://localhost:8000",
) -> list[str]:
    """Return a list of agent app names on an ADK server."""
    resp = requests.get(f"{host.rstrip('/')}/list-apps")
    resp.raise_for_status()
    return resp.json()

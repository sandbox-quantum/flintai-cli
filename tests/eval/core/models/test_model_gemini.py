import unittest
from unittest.mock import AsyncMock, MagicMock

from google.genai import types as genai_types

from flintai.eval.common.schema import Content, Message, PartType, Role
from flintai.eval.core.models.model import ResponseStatus
from flintai.eval.core.models.model_gemini import GeminiModel


def _make_response(text: str):
    response = MagicMock()
    content = genai_types.Content(
        role="model",
        parts=[genai_types.Part(text=text)],
    )
    candidate = MagicMock()
    candidate.content = content
    response.candidates = [candidate]
    return response


def _make_blocked_response():
    """Simulate a response blocked by safety filters."""
    response = MagicMock()
    response.candidates = None
    return response


def _make_blocked_response_empty_content():
    """Simulate a blocked response with a candidate but no content."""
    response = MagicMock()
    candidate = MagicMock()
    candidate.content = None
    response.candidates = [candidate]
    return response


class TestGeminiModel(unittest.IsolatedAsyncioTestCase):

    async def test_generate_text(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response("Hello!")
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.OK)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(
            resp.message.content.parts[0].text, "Hello!",
        )

    async def test_blocked_response_returns_none_message(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_blocked_response()
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "bad prompt"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_blocked_response_empty_content(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_blocked_response_empty_content()
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "bad prompt"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_blocked_with_safety_finish_reason(self):
        mock_client = MagicMock()
        response = MagicMock()
        candidate = MagicMock()
        candidate.content = None
        candidate.finish_reason = "SAFETY"
        response.candidates = [candidate]
        response.prompt_feedback = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "bad"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_blocked_with_recitation_finish_reason(self):
        mock_client = MagicMock()
        response = MagicMock()
        candidate = MagicMock()
        candidate.content = None
        candidate.finish_reason = "RECITATION"
        response.candidates = [candidate]
        response.prompt_feedback = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "copy"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(
            resp.status, ResponseStatus.BLOCKED_RECITATION,
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import AsyncMock, MagicMock

from anthropic.types import Message as AnthropicMessage, TextBlock, Usage

from flintai.eval.common.schema import Content, Message, PartType, Role
from flintai.eval.core.models.model_anthropic import AnthropicModel


def _make_response(text: str) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6-20250514",
        content=[TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


class TestAnthropicModel(unittest.IsolatedAsyncioTestCase):

    async def test_generate_text(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_response("Hello!"))

        model = AnthropicModel(client=mock_client, model="claude-sonnet-4-6-20250514")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(resp.message.content.parts[0].text, "Hello!")
        mock_client.messages.create.assert_called_once()

    async def test_generate_passes_kwargs(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_response("Hi"))

        model = AnthropicModel(client=mock_client, model="claude-sonnet-4-6-20250514")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, temperature=0.7)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.7)

    async def test_generate_system_message(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_response("OK"))

        model = AnthropicModel(client=mock_client, model="claude-sonnet-4-6-20250514")
        msg = Message(content=Content.text(Role.SYSTEM, "You are helpful."))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertIn("system", call_kwargs)

    async def test_max_tokens_default(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_response("Hi"))

        model = AnthropicModel(client=mock_client, model="claude-sonnet-4-6-20250514")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["max_tokens"], 1024)

    async def test_max_tokens_override(self):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_response("Hi"))

        model = AnthropicModel(client=mock_client, model="claude-sonnet-4-6-20250514", max_tokens=2048)
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, max_tokens=512)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["max_tokens"], 512)


if __name__ == "__main__":
    unittest.main()

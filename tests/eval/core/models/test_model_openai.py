import unittest
from unittest.mock import AsyncMock, MagicMock

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice

from flintai.eval.common.schema import Content, Message, PartType, Role
from flintai.eval.core.models.model_openai import OpenAIModel


def _make_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        created=1700000000,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            ),
        ],
    )


class TestOpenAIModel(unittest.IsolatedAsyncioTestCase):

    async def test_generate_text(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_make_completion("Hello!"))

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(resp.message.content.parts[0].text, "Hello!")
        mock_client.chat.completions.create.assert_called_once()

    async def test_generate_passes_kwargs(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_make_completion("Hi"))

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, temperature=0.5, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs.kwargs["max_tokens"], 100)

    async def test_generate_system_message(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_make_completion("OK"))

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.SYSTEM, "You are helpful."))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()

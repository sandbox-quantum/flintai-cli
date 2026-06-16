import asyncio
import unittest

from flintai.eval.common.schema import Content, Message, Role
from flintai.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
)


class StubModel(Model):
    """Concrete Model for testing the base class behavior."""

    def __init__(
        self,
        response: ModelResponse | None = None,
        error: Exception | None = None,
    ):
        self._response = response
        self._error = error

    async def _generate(self, messages, **kwargs):
        if self._error:
            raise self._error
        return self._response


class TestModelGenerate(unittest.TestCase):

    def test_generate_with_string_input(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        result = asyncio.run(model.generate("hello"))
        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(
            result.message.content.parts[0].text, "reply",
        )

    def test_generate_with_message_input(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        msg = Message(content=Content.text(Role.USER, "hi"))
        result = asyncio.run(model.generate(msg))
        self.assertEqual(result.status, ResponseStatus.OK)

    def test_generate_with_message_list(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        msgs = [Message(content=Content.text(Role.USER, "hi"))]
        result = asyncio.run(model.generate(msgs))
        self.assertEqual(result.status, ResponseStatus.OK)

    def test_generate_error_logs_and_reraises(self):
        error = ValueError("model failure")
        model = StubModel(error=error)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(model.generate("test"))
        self.assertIn("model failure", str(ctx.exception))

    def test_generate_non_ok_status(self):
        response = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "blocked"),
            ),
            status=ResponseStatus.BLOCKED_SAFETY,
        )
        model = StubModel(response=response)
        result = asyncio.run(model.generate("test"))
        self.assertEqual(result.status, ResponseStatus.BLOCKED_SAFETY)

    def test_generate_none_message(self):
        response = ModelResponse(message=None)
        model = StubModel(response=response)
        result = asyncio.run(model.generate("test"))
        self.assertIsNone(result.message)


if __name__ == "__main__":
    unittest.main()

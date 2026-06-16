import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from flintai.eval.common.schema import Content, Message, Role
from flintai.eval.core.models.model_adk import ADKModel


def _make_aiohttp_session_with_calls(call_responses: list[dict]):
    """Create a mock aiohttp session that returns different responses
    for sequential POST calls (e.g. create-session then run)."""
    responses = []
    for json_data in call_responses:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=json_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        responses.append(mock_response)

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=responses)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    return mock_session


class TestADKModel(unittest.IsolatedAsyncioTestCase):

    @patch("flintai.eval.core.models.model_adk.aiohttp.ClientSession")
    async def test_generate_text(self, mock_session_cls):
        mock_session = _make_aiohttp_session_with_calls([
            {"id": "session-1"},
            [
                {
                    "author": "my_agent",
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello!"}],
                    },
                },
            ],
        ])
        mock_session_cls.return_value = mock_session

        model = ADKModel(
            app_name="my_agent",
            host="http://localhost:8000",
        )
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(
            resp.message.content.role, Role.ASSISTANT,
        )
        self.assertEqual(
            resp.message.content.parts[0].text, "Hello!",
        )

    @patch("flintai.eval.core.models.model_adk.aiohttp.ClientSession")
    async def test_generate_empty_response(self, mock_session_cls):
        mock_session = _make_aiohttp_session_with_calls([
            {"id": "session-1"},
            [],
        ])
        mock_session_cls.return_value = mock_session

        model = ADKModel(
            app_name="my_agent",
            host="http://localhost:8000",
        )
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)

    @patch("flintai.eval.core.models.model_adk.aiohttp.ClientSession")
    async def test_new_session_per_call(self, mock_session_cls):
        mock_session1 = _make_aiohttp_session_with_calls([
            {"id": "session-1"},
            [
                {
                    "author": "my_agent",
                    "content": {
                        "role": "model",
                        "parts": [{"text": "First"}],
                    },
                },
            ],
        ])
        mock_session2 = _make_aiohttp_session_with_calls([
            {"id": "session-2"},
            [
                {
                    "author": "my_agent",
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Second"}],
                    },
                },
            ],
        ])
        mock_session_cls.side_effect = [mock_session1, mock_session2]

        model = ADKModel(
            app_name="my_agent",
            host="http://localhost:8000",
        )
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)
        await model.generate(msg)

        # Each generate creates a new ClientSession
        self.assertEqual(mock_session_cls.call_count, 2)

    @patch("flintai.eval.core.models.model_adk.aiohttp.ClientSession")
    async def test_run_url(self, mock_session_cls):
        mock_session = _make_aiohttp_session_with_calls([
            {"id": "session-1"},
            [
                {
                    "author": "test_app",
                    "content": {
                        "role": "model",
                        "parts": [{"text": "ok"}],
                    },
                },
            ],
        ])
        mock_session_cls.return_value = mock_session

        model = ADKModel(
            app_name="test_app",
            host="http://myhost:9000",
        )
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)

        # Second POST call is the /run call
        run_call = mock_session.post.call_args_list[1]
        self.assertEqual(
            run_call.args[0], "http://myhost:9000/run",
        )
        payload = run_call.kwargs["json"]
        self.assertEqual(payload["appName"], "test_app")


if __name__ == "__main__":
    unittest.main()

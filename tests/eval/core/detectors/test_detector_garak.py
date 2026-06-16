import unittest
from unittest.mock import MagicMock, patch

from garak.detectors.apikey import ApiKey

from flintai.eval.common.schema import Content, Message, Part, PartType, Role
from flintai.eval.core.detectors.detector_garak import GarakDetector, _extract_text, _create_conversation
from flintai.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(content=Content.text(Role.ASSISTANT, text))
    )


def _make_multipart_response(*texts: str) -> ModelResponse:
    parts = [Part.text_part(t) for t in texts]
    return ModelResponse(
        message=Message(content=Content(role=Role.ASSISTANT, parts=parts))
    )


class TestExtractText(unittest.TestCase):

    def test_single_text_part(self):
        response = _make_response("hello world")
        self.assertEqual(_extract_text(response), "hello world")

    def test_multiple_text_parts(self):
        response = _make_multipart_response("hello", "world")
        self.assertEqual(_extract_text(response), "hello\nworld")

    def test_none_message(self):
        response = ModelResponse(message=None)
        self.assertEqual(_extract_text(response), "")

    def test_non_text_parts_ignored(self):
        parts = [
            Part.text_part("visible"),
            Part.thinking_part("hidden thinking"),
        ]
        response = ModelResponse(
            message=Message(content=Content(role=Role.ASSISTANT, parts=parts))
        )
        self.assertEqual(_extract_text(response), "visible")


class TestCreateConversation(unittest.TestCase):

    def test_creates_turn_from_response(self):
        response = _make_response("test output")
        conversation = _create_conversation(response)

        self.assertEqual(len(conversation.turns), 1)
        self.assertEqual(conversation.turns[0].role, "assistant")
        self.assertEqual(conversation.turns[0].content.text, "test output")

    def test_empty_conversation_for_none_message(self):
        response = ModelResponse(message=None)
        conversation = _create_conversation(response)
        self.assertEqual(len(conversation.turns), 0)

    def test_user_role_mapping(self):
        response = ModelResponse(
            message=Message(content=Content.text(Role.USER, "user text"))
        )
        conversation = _create_conversation(response)
        self.assertEqual(conversation.turns[0].role, "user")

    def test_system_role_mapping(self):
        response = ModelResponse(
            message=Message(content=Content.text(Role.SYSTEM, "system text"))
        )
        conversation = _create_conversation(response)
        self.assertEqual(conversation.turns[0].role, "system")


class TestGarakDetector(unittest.IsolatedAsyncioTestCase):

    def _make_detector(self, mock_garak):
        """Create a GarakDetector with a pre-injected mock."""
        detector = GarakDetector("detectors.test.Test")
        detector._detector = mock_garak
        return detector

    async def test_no_hit_returns_1(self):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = [0.0, 0.0]

        detector = self._make_detector(mock_garak)
        result = await detector.detect(_make_response("safe output"))

        self.assertEqual(result.score, 1.0)
        mock_garak.detect.assert_called_once()

    async def test_full_hit_returns_0(self):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = [0.0, 1.0]

        detector = self._make_detector(mock_garak)
        result = await detector.detect(_make_response("leaked API key"))

        self.assertEqual(result.score, 0.0)

    async def test_partial_hit(self):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = [0.3, 0.6]

        detector = self._make_detector(mock_garak)
        result = await detector.detect(_make_response("borderline"))

        self.assertAlmostEqual(result.score, 0.4)

    async def test_none_results_returns_1(self):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = [None, None]

        detector = self._make_detector(mock_garak)
        result = await detector.detect(_make_response("output"))

        self.assertEqual(result.score, 1.0)

    async def test_exception_propagates(self):
        mock_garak = MagicMock()
        mock_garak.detect.side_effect = RuntimeError("garak failed")

        detector = self._make_detector(mock_garak)
        with self.assertRaises(RuntimeError):
            await detector.detect(_make_response("output"))

    async def test_empty_results_returns_1(self):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = []

        detector = self._make_detector(mock_garak)
        result = await detector.detect(_make_response("output"))

        self.assertEqual(result.score, 1.0)

    async def test_mixed_none_and_scores(self):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = [None, 0.2, None]

        detector = self._make_detector(mock_garak)
        result = await detector.detect(_make_response("output"))

        self.assertAlmostEqual(result.score, 0.8)

    @patch("garak._plugins.load_plugin")
    async def test_lazy_loads_plugin(self, mock_load_plugin):
        mock_garak = MagicMock()
        mock_garak.detect.return_value = [0.0]
        mock_load_plugin.return_value = mock_garak

        detector = GarakDetector("detectors.test.Test")
        await detector.detect(_make_response("output"))

        mock_load_plugin.assert_called_once_with(
            "detectors.test.Test"
        )


class TestGarakDetectorApiKeyE2E(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.detector = GarakDetector("detectors.apikey.ApiKey")

    async def test_detects_sendgrid_api_key(self):
        key = "SG.1234567890abcdefABCDEF.1234567890abcdefghijklmnopqrstuvwxyzABCDEFG"
        response = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, f"Here is your key: {key}")
            )
        )

        result = await self.detector.detect(response)

        self.assertEqual(result.score, 0.0)

    async def test_passes_message_without_key(self):
        response = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "I cannot generate API keys for you.")
            )
        )

        result = await self.detector.detect(response)

        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()

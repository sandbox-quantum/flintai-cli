import unittest
from unittest.mock import AsyncMock, MagicMock

from flintai.eval.common.schema import Content, Message, Role
from flintai.eval.core.detectors.detector_model import (
    ModelDetector,
    _extract_text,
    _parse_score,
)
from flintai.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(content=Content.text(Role.ASSISTANT, text))
    )


class TestExtractText(unittest.TestCase):

    def test_extracts_text(self):
        response = _make_response("hello")
        self.assertEqual(_extract_text(response), "hello")

    def test_none_message(self):
        response = ModelResponse(message=None)
        self.assertEqual(_extract_text(response), "")


class TestParseScore(unittest.TestCase):

    def test_parses_clean_score(self):
        self.assertEqual(_parse_score(_make_response("0.8")), 0.8)

    def test_parses_one(self):
        self.assertEqual(_parse_score(_make_response("1.0")), 1.0)

    def test_parses_zero(self):
        self.assertEqual(_parse_score(_make_response("0.0")), 0.0)

    def test_parses_integer_one(self):
        self.assertEqual(_parse_score(_make_response("1")), 1.0)

    def test_parses_integer_zero(self):
        self.assertEqual(_parse_score(_make_response("0")), 0.0)

    def test_parses_score_with_surrounding_text(self):
        response = _make_response("The score is 0.7 based on analysis")
        self.assertEqual(_parse_score(response), 0.7)

    def test_returns_none_for_unparseable(self):
        self.assertIsNone(
            _parse_score(_make_response("unsafe")),
        )

    def test_returns_none_for_none_message(self):
        response = ModelResponse(message=None)
        self.assertIsNone(_parse_score(response))


class TestModelDetector(unittest.IsolatedAsyncioTestCase):

    def _make_detector(self, score_text="1.0", prompt=None):
        model = MagicMock()
        model.generate = AsyncMock(return_value=_make_response(score_text))
        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        detector = ModelDetector(model=model, **kwargs)
        return detector, model

    async def test_safe_response(self):
        detector, _ = self._make_detector("1.0")
        result = await detector.detect(_make_response("safe text"))
        self.assertEqual(result.score, 1.0)

    async def test_unsafe_response(self):
        detector, _ = self._make_detector("0.0")
        result = await detector.detect(_make_response("unsafe text"))
        self.assertEqual(result.score, 0.0)

    async def test_partial_score(self):
        detector, _ = self._make_detector("0.6")
        result = await detector.detect(_make_response("borderline"))
        self.assertAlmostEqual(result.score, 0.6)

    async def test_passes_response_text_to_model(self):
        detector, model = self._make_detector("1.0")
        await detector.detect(_make_response("test content"))
        call_args = model.generate.call_args[0][0]
        prompt_text = call_args.content.parts[0].text
        self.assertIn("test content", prompt_text)

    async def test_custom_prompt(self):
        detector, model = self._make_detector(
            "0.5", prompt="Rate toxicity:",
        )
        await detector.detect(_make_response("some text"))
        call_args = model.generate.call_args[0][0]
        prompt_text = call_args.content.parts[0].text
        self.assertIn("Rate toxicity:", prompt_text)

    async def test_unparseable_model_output_returns_zero(self):
        detector, _ = self._make_detector(
            "I think it's somewhat safe",
        )
        result = await detector.detect(_make_response("text"))
        self.assertEqual(result.score, 0.0)

    async def test_zero_score_is_not_a_parse_failure(self):
        detector, _ = self._make_detector("0.0")
        with self.assertNoLogs(
            "flintai.eval.core.detectors.detector_model", level="WARNING",
        ):
            result = await detector.detect(_make_response("unsafe text"))
        self.assertEqual(result.score, 0.0)

    async def test_model_exception_propagates(self):
        model = MagicMock()
        model.generate = AsyncMock(side_effect=RuntimeError("model error"))
        detector = ModelDetector(model=model)
        with self.assertRaises(RuntimeError):
            await detector.detect(_make_response("text"))


if __name__ == "__main__":
    unittest.main()

import unittest

from flintai.eval.common.schema import Content, Message, Role
from flintai.eval.core.detectors.detector import DetectorResult
from flintai.eval.core.detectors.detector_secret import (
    SecretDetector,
    _extract_text,
)
from flintai.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(content=Content.text(Role.ASSISTANT, text)),
    )


class TestSecretDetector(unittest.IsolatedAsyncioTestCase):

    async def test_detect_clean_response(self):
        detector = SecretDetector()
        response = _make_response(
            "The weather today is sunny and warm.",
        )
        result = await detector.detect(response)

        self.assertIsInstance(result, DetectorResult)
        self.assertAlmostEqual(result.score, 1.0)

    async def test_detect_aws_key(self):
        detector = SecretDetector()
        response = _make_response(
            "Your AWS access key is "
            "AKIAIOSFODNN7EXAMPLE",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_private_key(self):
        detector = SecretDetector()
        response = _make_response(
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn\n"
            "-----END RSA PRIVATE KEY-----",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_stripe_key(self):
        detector = SecretDetector()
        response = _make_response(
            "Use this Stripe key: "
            "sk_live_abc123def456ghi789jkl012mno",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_none_message(self):
        detector = SecretDetector()
        response = ModelResponse(message=None)
        result = await detector.detect(response)

        self.assertAlmostEqual(result.score, 1.0)

    async def test_score_floors_at_zero(self):
        detector = SecretDetector()
        response = _make_response(
            "AKIAIOSFODNN7EXAMPLE\n"
            "sk_live_abc123def456ghi789jkl012mno\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "-----BEGIN DSA PRIVATE KEY-----\n"
            "-----BEGIN EC PRIVATE KEY-----\n"
            "-----BEGIN PGP PRIVATE KEY BLOCK-----",
        )
        result = await detector.detect(response)

        self.assertAlmostEqual(result.score, 0.0)


class TestExtractText(unittest.TestCase):

    def test_extracts_text(self):
        response = _make_response("hello world")
        self.assertEqual(_extract_text(response), "hello world")

    def test_returns_empty_for_none(self):
        response = ModelResponse(message=None)
        self.assertEqual(_extract_text(response), "")


if __name__ == "__main__":
    unittest.main()

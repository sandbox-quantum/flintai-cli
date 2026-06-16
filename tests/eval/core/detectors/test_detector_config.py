import unittest
from unittest.mock import MagicMock, patch

from flintai.eval.db.base.detectors.detector_types import DbDetector, DetectorType


class TestDbDetector(unittest.TestCase):

    def test_garak_roundtrip(self):
        config = DbDetector(
            type=DetectorType.GARAK,
            name="mitigation-bypass",
            detector_name="detectors.mitigation.MitigationBypass",
        )
        data = config.to_dict()
        restored = DbDetector.from_dict(data)
        self.assertEqual(restored.type, DetectorType.GARAK)
        self.assertEqual(restored.name, "mitigation-bypass")
        self.assertEqual(
            restored.detector_name,
            "detectors.mitigation.MitigationBypass",
        )

    def test_topic_guard_roundtrip(self):
        config = DbDetector(
            type=DetectorType.TOPIC_GUARD,
            name="topic-guard-test",
            agent_objective="Help users book flights",
            agent_instructions="Only discuss travel",
        )
        data = config.to_dict()
        restored = DbDetector.from_dict(data)
        self.assertEqual(restored.type, DetectorType.TOPIC_GUARD)
        self.assertEqual(restored.name, "topic-guard-test")
        self.assertEqual(
            restored.agent_objective,
            "Help users book flights",
        )
        self.assertEqual(
            restored.agent_instructions,
            "Only discuss travel",
        )

    def test_topic_guard_objective_only(self):
        config = DbDetector(
            type=DetectorType.TOPIC_GUARD,
            name="tg-obj",
            agent_objective="Book flights",
        )
        data = config.to_dict()
        restored = DbDetector.from_dict(data)
        self.assertEqual(
            restored.agent_objective, "Book flights",
        )
        self.assertIsNone(restored.agent_instructions)

    def test_topic_guard_instructions_only(self):
        config = DbDetector(
            type=DetectorType.TOPIC_GUARD,
            name="tg-instr",
            agent_instructions="Only travel topics",
        )
        data = config.to_dict()
        restored = DbDetector.from_dict(data)
        self.assertIsNone(restored.agent_objective)
        self.assertEqual(
            restored.agent_instructions,
            "Only travel topics",
        )


class TestCreateDetector(unittest.TestCase):

    def test_create_garak_detector(self):
        from flintai.eval.core.detectors.detector_garak import GarakDetector
        from flintai.eval.db.base.detectors.detector_helpers import create_detector

        config = DbDetector(
            type=DetectorType.GARAK,
            name="test-det",
            detector_name="detectors.mitigation.MitigationBypass",
        )
        detector = create_detector(config)

        self.assertIsInstance(detector, GarakDetector)
        self.assertEqual(
            detector._detector_name,
            "detectors.mitigation.MitigationBypass",
        )

    def test_create_garak_detector_missing_name(self):
        from flintai.eval.db.base.detectors.detector_helpers import create_detector

        config = DbDetector(
            type=DetectorType.GARAK,
            name="test-det",
        )
        with self.assertRaises(ValueError):
            create_detector(config)

    @patch(
        "flintai.eval.core.models.generator_model"
        ".get_generator_model",
    )
    def test_create_topic_guard_detector(
        self, mock_get_model,
    ):
        from flintai.eval.core.detectors.detector_topic_guard import (
            TopicGuardDetector,
        )
        from flintai.eval.db.base.detectors.detector_helpers import (
            create_detector,
        )

        mock_get_model.return_value = MagicMock()
        config = DbDetector(
            type=DetectorType.TOPIC_GUARD,
            name="tg-test",
            agent_objective="Book flights",
        )
        detector = create_detector(config)
        self.assertIsInstance(detector, TopicGuardDetector)
        self.assertEqual(
            detector.agent_objective, "Book flights",
        )

    def test_create_topic_guard_detector_missing_params(self):
        from flintai.eval.db.base.detectors.detector_helpers import (
            create_detector,
        )

        config = DbDetector(
            type=DetectorType.TOPIC_GUARD,
            name="tg-empty",
        )
        with self.assertRaises(ValueError):
            create_detector(config)


if __name__ == "__main__":
    unittest.main()

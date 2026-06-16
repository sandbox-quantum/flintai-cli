import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from flintai.eval.common.schema import Content, Message, Role
from flintai.eval.core.detectors.detector import DetectorResult
from flintai.eval.core.eval.evaluation import EvaluationStatus
from flintai.eval.core.eval.evaluation_garak_module import GarakModuleEvaluation
from flintai.eval.core.eval.evaluation_garak_probe import GarakProbeEvaluation
from flintai.eval.core.models.model import ModelResponse, ResponseStatus


class TestGarakModuleEvaluation(unittest.IsolatedAsyncioTestCase):
    @patch("flintai.eval.core.eval.evaluation_garak_module.enumerate_plugins")
    @patch("flintai.eval.core.eval.evaluation_garak_probe._plugins")
    async def test_init_creates_probe_children(
        self, mock_probe_plugins, mock_enumerate,
    ):
        mock_enumerate.return_value = [
            ("probes.mymod.ProbeA", True),
            ("probes.mymod.ProbeB", True),
            ("probes.mymod.Inactive", False),
            ("probes.other.ProbeC", True),
        ]

        probe = MagicMock()
        probe.prompts = ["p1"]
        probe.primary_detector = "always.Pass"
        mock_probe_plugins.load_plugin.return_value = probe

        e = GarakModuleEvaluation(module_name="mymod")
        await e.init()

        self.assertEqual(e.status, EvaluationStatus.INITIALIZED)
        self.assertEqual(len(e.children), 2)
        for child in e.children:
            self.assertIsInstance(child, GarakProbeEvaluation)
        names = [c.probe_name for c in e.children]
        self.assertIn("probes.mymod.ProbeA", names)
        self.assertIn("probes.mymod.ProbeB", names)

    @patch("flintai.eval.core.eval.evaluation_garak_module.enumerate_plugins")
    @patch("flintai.eval.core.eval.evaluation_garak_probe._plugins")
    async def test_run_executes_all_probes(
        self, mock_probe_plugins, mock_enumerate,
    ):
        mock_enumerate.return_value = [
            ("probes.mod.ProbeA", True),
        ]

        probe = MagicMock()
        probe.prompts = ["prompt"]
        probe.primary_detector = "always.Pass"
        mock_probe_plugins.load_plugin.return_value = probe

        e = GarakModuleEvaluation(module_name="mod")
        await e.init()

        model = AsyncMock()
        response_msg = Message(
            content=Content.text(Role.ASSISTANT, "ok"),
        )
        model.generate = AsyncMock(
            return_value=ModelResponse(
                message=response_msg, status=ResponseStatus.OK,
            ),
        )

        # Patch detector on the leaf SinglePromptEvaluation
        probe_eval = e.children[0]
        leaf = probe_eval.children[0]
        detector = AsyncMock()
        detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.7),
        )
        leaf.detector = detector

        await e.run(model, concurrency=4)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        summary = e.get_summary()
        self.assertAlmostEqual(summary.achieved_score, 0.7)

    @patch("flintai.eval.core.eval.evaluation_garak_module.enumerate_plugins")
    async def test_no_probes_found_errors(self, mock_enumerate):
        mock_enumerate.return_value = [
            ("probes.other.Probe", True),
        ]

        e = GarakModuleEvaluation(module_name="nonexistent")
        await e.init()

        self.assertEqual(e.status, EvaluationStatus.ERROR)
        self.assertIn("no active probes", e.error_message)

    def test_missing_module_name_errors(self):
        with self.assertRaises(TypeError):
            GarakModuleEvaluation()


if __name__ == "__main__":
    unittest.main()

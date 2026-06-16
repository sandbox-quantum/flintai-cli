import unittest
from unittest.mock import patch, MagicMock

from flintai.eval.core.message.message_collection_garak import (
    GarakMessageCollection,
)


class TestGarakMessageCollection(unittest.TestCase):

    @patch(
        "flintai.eval.core.message.message_collection_garak._plugins"
    )
    def test_load_returns_messages(self, mock_plugins):
        mock_probe = MagicMock()
        mock_probe.prompts = ["prompt 1", "prompt 2"]
        mock_plugins.load_plugin.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        messages = collection.load()

        self.assertEqual(len(messages), 2)
        self.assertEqual(
            messages[0].content.parts[0].text, "prompt 1",
        )
        mock_plugins.load_plugin.assert_called_once_with(
            "probes.test.Probe",
        )

    @patch(
        "flintai.eval.core.message.message_collection_garak._plugins"
    )
    def test_load_caches_messages(self, mock_plugins):
        mock_probe = MagicMock()
        mock_probe.prompts = ["prompt"]
        mock_plugins.load_plugin.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        collection.load()
        collection.load()

        mock_plugins.load_plugin.assert_called_once()

    @patch(
        "flintai.eval.core.message.message_collection_garak._plugins"
    )
    def test_get_existing(self, mock_plugins):
        mock_probe = MagicMock()
        mock_probe.prompts = ["prompt"]
        mock_plugins.load_plugin.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        messages = collection.load()
        msg = collection.get(messages[0].id)

        self.assertEqual(
            msg.content.parts[0].text, "prompt",
        )

    @patch(
        "flintai.eval.core.message.message_collection_garak._plugins"
    )
    def test_get_missing_raises(self, mock_plugins):
        mock_probe = MagicMock()
        mock_probe.prompts = []
        mock_plugins.load_plugin.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        with self.assertRaises(KeyError):
            collection.get("no-such-id")

    @patch(
        "flintai.eval.core.message.message_collection_garak._plugins"
    )
    def test_save_raises(self, mock_plugins):
        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        with self.assertRaises(NotImplementedError):
            collection.save([])

    @patch(
        "flintai.eval.core.message.message_collection_garak._plugins"
    )
    def test_size_returns_count(self, mock_plugins):
        mock_probe = MagicMock()
        mock_probe.prompts = ["a", "b", "c"]
        mock_plugins.load_plugin.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        self.assertEqual(collection.size(), 3)


if __name__ == "__main__":
    unittest.main()

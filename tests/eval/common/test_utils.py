import unittest
from unittest.mock import patch

from flintai.eval.common.utils import resolve_env, resolve_env_dict


class TestResolveEnv(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(resolve_env(None))

    def test_plain_string_unchanged(self):
        self.assertEqual(resolve_env("sk-abc123"), "sk-abc123")

    def test_empty_string_unchanged(self):
        self.assertEqual(resolve_env(""), "")

    @patch.dict("os.environ", {"MY_KEY": "resolved-value"})
    def test_full_variable_replacement(self):
        self.assertEqual(
            resolve_env("${MY_KEY}"), "resolved-value",
        )

    @patch.dict("os.environ", {"TOKEN": "abc123"})
    def test_inline_interpolation(self):
        self.assertEqual(
            resolve_env("Bearer ${TOKEN}"),
            "Bearer abc123",
        )

    @patch.dict(
        "os.environ",
        {"HOST": "example.com", "PORT": "8080"},
    )
    def test_multiple_variables(self):
        self.assertEqual(
            resolve_env("https://${HOST}:${PORT}"),
            "https://example.com:8080",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_variable_raises(self):
        with self.assertRaises(ValueError, msg="MISSING_VAR"):
            resolve_env("${MISSING_VAR}")

    @patch.dict(
        "os.environ", {"FIRST": "one"}, clear=True,
    )
    def test_missing_second_variable_raises(self):
        with self.assertRaises(ValueError, msg="SECOND"):
            resolve_env("${FIRST}-${SECOND}")


class TestResolveEnvDict(unittest.TestCase):

    @patch.dict("os.environ", {"TOKEN": "secret"})
    def test_resolves_values(self):
        result = resolve_env_dict({
            "Authorization": "Bearer ${TOKEN}",
            "Accept": "application/json",
        })
        self.assertEqual(result, {
            "Authorization": "Bearer secret",
            "Accept": "application/json",
        })

    def test_empty_dict(self):
        self.assertEqual(resolve_env_dict({}), {})


if __name__ == "__main__":
    unittest.main()

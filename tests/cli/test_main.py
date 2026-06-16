from unittest.mock import patch

from flintai.cli.main import _print_error


class TestPrintError:
    def test_displays_error_type_and_message(self, capsys):
        error = ValueError("something went wrong")

        _print_error(error)

        captured = capsys.readouterr().out
        assert "ValueError" in captured
        assert "something went wrong" in captured

    def test_empty_message_shows_fallback(self, capsys):
        error = RuntimeError()

        _print_error(error)

        captured = capsys.readouterr().out
        assert "(no message)" in captured


class TestMainErrorHandling:
    @patch("flintai.cli.main._dispatch", side_effect=RuntimeError("test crash"))
    @patch("flintai.cli.main.setup_file_logging")
    @patch("flintai.cli.main._print_logo")
    @patch("flintai.cli.main.load_dotenv")
    @patch("flintai.cli.main.init_cli")
    def test_exception_prints_error_and_continues(
        self, mock_init_cli, mock_dotenv, mock_logo,
        mock_logging, mock_dispatch,
        capsys, tmp_path,
    ):
        mock_init_cli.get_flintai_env_path.return_value = tmp_path / "nonexistent"

        from flintai.cli.main import main
        main(["eval", "models", "list"])

        captured = capsys.readouterr().out
        assert "RuntimeError" in captured
        assert "test crash" in captured
        assert "Completed in" in captured

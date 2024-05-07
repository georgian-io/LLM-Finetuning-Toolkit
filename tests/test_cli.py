from unittest.mock import patch

from typer.testing import CliRunner

from llmtune.cli.toolkit import app, cli


runner = CliRunner()


def test_run_command():
    # Test the `run` command
    with patch("llmtune.cli.toolkit.run_one_experiment") as mock_run_one_experiment:
        result = runner.invoke(app, ["run", "./llmtune/config.yml"])
        assert result.exit_code == 0
        mock_run_one_experiment.assert_called_once()


def test_generate_config_command():
    # Test the `generate config` command
    with patch("llmtune.cli.toolkit.shutil.copy") as mock_copy:
        result = runner.invoke(app, ["generate", "config"])
        assert result.exit_code == 0
        mock_copy.assert_called_once()


def test_cli():
    # Test the `cli` function
    with patch("llmtune.cli.toolkit.app") as mock_app:
        cli()
        mock_app.assert_called_once()

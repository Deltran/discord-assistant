"""Tests for the main entry point."""

from unittest.mock import patch

import pytest


def test_main_creates_bot():
    with patch.dict("os.environ", {
        "DISCORD_TOKEN": "test-token",
        "MINIMAX_API_KEY": "test-key",
        "MONITORING_CHANNEL_ID": "1",
    }):
        from importlib import reload
        import src.settings
        reload(src.settings)
        from src.main import create_app
        bot = create_app()
        assert bot is not None
        assert bot.settings.discord_token.get_secret_value() == "test-token"
        assert bot._agent_callback is not None

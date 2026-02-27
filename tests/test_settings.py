"""Tests for application settings."""

import os
from unittest.mock import patch


def test_settings_loads_from_env():
    env = {
        "DISCORD_TOKEN": "test-token",
        "MINIMAX_API_KEY": "test-key",
        "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        "MINIMAX_MODEL": "MiniMax-M2.5",
        "MONITORING_CHANNEL_ID": "123456",
        "ASSISTANT_HOME": "/tmp/test-assistant",
    }
    with patch.dict(os.environ, env, clear=False):
        from importlib import reload
        import src.settings as settings_mod
        reload(settings_mod)
        s = settings_mod.Settings()
        assert s.discord_token.get_secret_value() == "test-token"
        assert s.minimax_api_key.get_secret_value() == "test-key"
        assert s.minimax_base_url == "https://api.minimax.io/v1"
        assert s.minimax_model == "MiniMax-M2.5"
        assert s.monitoring_channel_id == 123456


def test_settings_defaults():
    env = {
        "DISCORD_TOKEN": "t",
        "MINIMAX_API_KEY": "k",
        "MONITORING_CHANNEL_ID": "1",
    }
    with patch.dict(os.environ, env, clear=False):
        from importlib import reload
        import src.settings as settings_mod
        reload(settings_mod)
        s = settings_mod.Settings()
        assert s.minimax_base_url == "https://api.minimax.io/v1"
        assert s.minimax_model == "MiniMax-M2.5"


def test_settings_path_properties():
    env = {
        "DISCORD_TOKEN": "t",
        "MINIMAX_API_KEY": "k",
        "MONITORING_CHANNEL_ID": "1",
        "ASSISTANT_HOME": "/tmp/test-home",
    }
    with patch.dict(os.environ, env, clear=False):
        from importlib import reload
        import src.settings as settings_mod
        reload(settings_mod)
        s = settings_mod.Settings()
        assert str(s.soul_path) == "/tmp/test-home/SOUL.md"
        assert str(s.memory_dir) == "/tmp/test-home/memory"
        assert str(s.skills_dir) == "/tmp/test-home/skills"
        assert str(s.data_dir) == "/tmp/test-home/data"
        assert str(s.log_dir) == "/tmp/test-home/logs"

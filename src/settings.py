"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    discord_token: SecretStr
    minimax_api_key: SecretStr
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_model: str = "MiniMax-M2.5"
    monitoring_channel_id: int = 0
    assistant_home: Path = Path.home() / ".assistant"

    @property
    def soul_path(self) -> Path:
        return self.assistant_home / "SOUL.md"

    @property
    def memory_dir(self) -> Path:
        return self.assistant_home / "memory"

    @property
    def skills_dir(self) -> Path:
        return self.assistant_home / "skills"

    @property
    def data_dir(self) -> Path:
        return self.assistant_home / "data"

    @property
    def log_dir(self) -> Path:
        return self.assistant_home / "logs"

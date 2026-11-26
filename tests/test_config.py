import pytest
from pydantic import ValidationError

from support_agent.platform.config import Settings


def test_database_url_is_required_without_environment_or_dotenv(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError) as caught:
        Settings(
            _env_file=None,
            openai_base_url="https://upstream.invalid/v1",
            openai_api_key="test-key",
            openai_model="test-model",
        )
    assert any(error["loc"] == ("database_url",) for error in caught.value.errors())


def test_database_url_loads_dotenv_and_environment_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    dotenv_url = "mysql+aiomysql://demo:example@127.0.0.1:3306/dotenv_db?charset=utf8mb4"
    environment_url = "mysql+aiomysql://demo:example@127.0.0.1:3307/environment_db?charset=utf8mb4"
    (tmp_path / ".env").write_text(f"DATABASE_URL={dotenv_url}\n", encoding="utf-8")
    model_config = {
        "openai_base_url": "https://upstream.invalid/v1",
        "openai_api_key": "test-key",
        "openai_model": "test-model",
    }
    assert Settings(**model_config).database_url.get_secret_value() == dotenv_url
    monkeypatch.setenv("DATABASE_URL", environment_url)
    settings = Settings(**model_config)
    assert settings.database_url.get_secret_value() == environment_url
    assert "example@" not in repr(settings)

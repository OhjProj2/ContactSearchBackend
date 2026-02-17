from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_PATH = Path(__file__).parent.parent / "env" / ".env"


class Settings(BaseSettings):
    OLLAMA_URL: str
    OLLAMA_PORT: str
    OLLAMA_USERNAME: str
    OLLAMA_PASSWORD: str
    OLLAMA_MODEL: str
    OLLAMA_TEMPERATURE: float
    OLLAMA_TOP_P: float
    OLLAMA_NUM_PREDICT: int
    OLLAMA_NUM_CTX: int

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")


def main():
    settings = Settings()
    print(settings)


if __name__ == "__main__":
    main()

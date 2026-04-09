from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_PATH = Path(__file__).parent.parent / "env" / ".env"


class Settings(BaseSettings):
    """Main Pydantic settings class using env vars from env/.env

    Attributes:
        OLLAMA_URL: Ollama server URL
        OLLAMA_PORT: Ollama server port
        OLLAMA_USERNAME: Ollama username for authentication
        OLLAMA_PASSWORD: Ollama password for authentication
        OLLAMA_MODEL: Model name
        OLLAMA_TEMPERATURE: Default sampling temperature (0.0-1.0)
        OLLAMA_TOP_P: Default nucleus sampling probability (0.0-1.0)
        OLLAMA_NUM_PREDICT: Default maximum tokens to predict
        OLLAMA_NUM_CTX: Default context window size
        OLLAMA_REPEAT_PENALTY: Default repeat penalty
        MONGODB_URI: Default MongoDB Universal Resource Identifier
        MONGODB_NAME: Default MongoDB database name (one db per one project)
        MONGODB_COLLECTION: Default MongoDB collection name (one collection per one kind of contact details)
        ADMIN_USERNAME: Frontend access login username
        ADMIN_PASSWORD: Frontend access login password
    """

    OLLAMA_URL: str
    OLLAMA_PORT: str
    OLLAMA_USERNAME: str
    OLLAMA_PASSWORD: str
    OLLAMA_MODEL: str
    OLLAMA_TEMPERATURE: float
    OLLAMA_TOP_P: float
    OLLAMA_NUM_PREDICT: int
    OLLAMA_NUM_CTX: int
    OLLAMA_REPEAT_PENALTY: int
    MONGODB_URI: str
    MONGODB_NAME: str
    MONGODB_COLLECTION: str
    ADMIN_USERNAME: str
    ADMIN_PASSWORD: str

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")


def main():
    settings = Settings()
    print(settings)


if __name__ == "__main__":
    main()

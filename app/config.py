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
        OLLAMA_TEMPERATURE: Sampling temperature (0.0-1.0)
        OLLAMA_TOP_P: Nucleus sampling probability (0.0-1.0)
        OLLAMA_NUM_PREDICT: Maximum tokens to predict
        OLLAMA_NUM_CTX: Context window size
        MONGODB_URI: MongoDB Universal Resource Identifier
        MONGODB_NAME: MongoDB database name (one db per one project)
        MONGODB_COLLECTION: MongoDB collection name (one collection per one kind of contact details)
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
    MONGODB_URI: str
    MONGODB_NAME: str
    MONGODB_COLLECTION: str

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")


def main():
    settings = Settings()
    print(settings)


if __name__ == "__main__":
    main()

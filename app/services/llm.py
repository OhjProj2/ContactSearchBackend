from langchain_ollama import ChatOllama

from config import Settings

settings = Settings()

def build_ollama_instance(
    model: str | None = None, temp: float | None = None , top_p: float | None = None, num_predict: int | None = None, num_ctx: int | None = None
) -> ChatOllama:
    return ChatOllama(
        base_url=f"https://{settings.OLLAMA_USERNAME}:{settings.OLLAMA_PASSWORD}@{settings.OLLAMA_URL}:{settings.OLLAMA_PORT}",
        model=model if model else settings.OLLAMA_MODEL,
        format="json",
        temperature=temp if temp else settings.OLLAMA_TEMPERATURE,
        top_p=top_p if top_p else settings.OLLAMA_TOP_P,
        num_predict=num_predict if num_predict else settings.OLLAMA_NUM_PREDICT,
        num_ctx=num_ctx if num_ctx else settings.OLLAMA_NUM_CTX,
        # client_kwargs are given straight to httpx client
        # client_kwargs={"verify": False},  # uncomment if using private SSL cert
    )

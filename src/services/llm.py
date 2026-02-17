from langchain_ollama import ChatOllama

from config import Settings

settings = Settings()


def build_ollama_instance(
    model: str, temp: float, top_p: float, num_predict: int, num_ctx: int
) -> ChatOllama:
    return ChatOllama(
        base_url=f"https://{settings.OLLAMA_USERNAME}:{settings.OLLAMA_PASSWORD}@{settings.OLLAMA_URL}:{settings.OLLAMA_URL}",
        model=model if model else settings.OLLAMA_MODEL,
        format="json",
        temperature=temp if temp else settings.OLLAMA_TEMPERATURE,
        top_p=top_p if top_p else settings.OLLAMA_TOP_P,
        num_predict=num_predict if num_predict else settings.OLLAMA_NUM_PREDICT,
        num_ctx=num_ctx if num_ctx else settings.OLLAMA_NUM_CTX,
        # client_kwargs are given straight to httpx client
        # client_kwargs={"verify": False},  # uncomment if using private SSL cert
    )

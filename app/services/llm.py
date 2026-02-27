from langchain_ollama import ChatOllama

from config import Settings

settings = Settings()


def build_ollama_instance(
    model: str, temp: float, top_p: float, num_predict: int, num_ctx: int
) -> ChatOllama:
    return ChatOllama(
        base_url=f"https://{settings.OLLAMA_USERNAME}:{settings.OLLAMA_PASSWORD}@{settings.OLLAMA_URL}:{settings.OLLAMA_PORT}",
        model=model,
        format="json",
        temperature=temp,
        top_p=top_p,
        num_predict=num_predict,
        num_ctx=num_ctx,
        # client_kwargs are given straight to httpx client
        # client_kwargs={"verify": False},  # uncomment if using private SSL cert
    )

from langchain_ollama import ChatOllama

from config import Settings

settings = Settings()


def build_ollama_instance(
    model: str, temp: float, top_p: float, num_predict: int, num_ctx: int
) -> ChatOllama:
    """Builds a ChatOllama instance based on model settings and auth credentials.

    Args:
        model: model name of the Ollama model
        temp: sampling temperature (0.0-1.0)
        top_p: nucleus sampling (0.0-1.0)
        num_predict: max number of tokens to predict
        num_ctx: context window size (in tokens)

    Returns:
        ChatOllama instance that does the communication with Ollama LLMs
    """
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

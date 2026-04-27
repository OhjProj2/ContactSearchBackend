from langchain_ollama import ChatOllama
import httpx
from app.config import Settings

settings = Settings()


def build_ollama_instance(
    model: str,
    temp: float,
    top_p: float,
    num_predict: int,
    num_ctx: int,
    repeat_penalty: float,
    timeout: int,
) -> ChatOllama:
    """Builds a ChatOllama instance based on model settings and auth credentials.

    Args:
        model: model name of the Ollama model
        temp: sampling temperature (0.0-1.0)
        top_p: nucleus sampling (0.0-1.0)
        num_predict: max number of tokens to predict
        num_ctx: context window size (in tokens)
        repeat_penalty: scales probability of tokens already appeared
        timeout: time in seconds until langchain raises timeout error

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
        repeat_penalty=repeat_penalty,
        # client_kwargs are given straight to httpx client
        # client_kwargs={"verify": False},  # uncomment if using private SSL cert
        client_kwargs={
            "timeout": httpx.Timeout(60.0, connect=10.0, read=timeout, write=10.0)
        },
    )

import os
from pathlib import Path
import asyncio
from typing import List

from langchain_ollama import ChatOllama
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler

env_path = Path.cwd() / ".." / "env" / ".env"
load_dotenv(dotenv_path=env_path)
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_USERNAME = os.getenv("OLLAMA_USERNAME")
OLLAMA_PASSWORD = os.getenv("OLLAMA_PASSWORD")


class SocialMedia(BaseModel):
    linkedin: str
    twitter_x: str
    facebook: str
    telegram: str
    signal: str
    instagram: str


class ContactInfo(BaseModel):
    occupation: str
    occupation_role: str
    organization: str
    first_name: str
    last_name: str
    email: str
    phone: str
    mobile_phone: str
    street_address: str
    postal_code: str
    city: str
    country: str
    web_page: str
    social_media: SocialMedia
    business_id: str
    company_name: str
    ngo_name: str
    political_party: str


class ContactList(BaseModel):
    contacts: List[ContactInfo]


models_path = Path.cwd() / ".." / "env" / "models.txt"
model_list = [model for model in sorted(models_path.read_text().splitlines())]

system_message_path = Path.cwd() / ".." / "env" / "system_message.txt"
system_message = system_message_path.read_text()

url_list_path = Path.cwd() / ".." / "env" / "url_list.txt"
url_list = [url for url in sorted(url_list_path.read_text().splitlines())]


async def fetch_web_page(target_url):
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(target_url)


if __name__ == "__main__":
    print("=" * 25)
    print("LIST OF MODELS")
    print("=" * 25)
    for i, model in enumerate(model_list, start=1):
        print(f"{i}. {model}")
    print("=" * 25)
    choice_model = model_list[int(input("Choose a model: ")) - 1]
    print(f"You chose: {choice_model}")
    print("=" * 25)
    print("LIST OF MODELS")
    print("=" * 25)
    for i, url in enumerate(url_list, start=1):
        print(f"{i}. {url}")
    print("=" * 25)
    url = input("Give a URL or choose from the list:  ")
    try:
        choice_url = url_list[int(url) - 1]
    except ValueError:
        choice_url = url
    print(f"You gave/selected: {choice_url}")

    result = asyncio.run(fetch_web_page(choice_url))
    if not result.success:
        print(
            f"Failed to fetch URL: {result.status_code if hasattr(result, 'status_code') else 'unknown'}"
        )
        exit(1)

    markdown_content = result.markdown
    print(
        f"DEBUG: Fetched {len(result.html if hasattr(result, 'html') else '')} characters from URL."
    )
    print(f"DEBUG: Markdown content length: {len(markdown_content)}")

    user_prompt = f"Here is the content of a webpage:\n\n{markdown_content}"

    model = ChatOllama(
        base_url=f"https://{OLLAMA_USERNAME}:{OLLAMA_PASSWORD}@{OLLAMA_URL}:{OLLAMA_PORT}",
        model=choice_model,
        format="json",
        temperature=0.0,
        top_p=0.5,
        num_predict=65536,
        num_ctx=65536,
        # client_kwargs are given straight to httpx client
        client_kwargs={"verify": False},  # because of private SSL cert
    )

    # JSON must be mentioned in prompt even when 'format="json"' is given as parameter
    messages = [
        (
            "system",
            system_message,
        ),
        ("human", user_prompt),
    ]

    structured_model = model.with_structured_output(ContactList)
    result = structured_model.invoke(messages)
    print(result)

    # print(model.invoke(messages))

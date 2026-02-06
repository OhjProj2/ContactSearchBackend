import os
from pathlib import Path
import asyncio
from typing import List

from langchain_ollama import ChatOllama
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from crawl4ai import AsyncWebCrawler

env_path = Path.cwd() / ".." / "env" / ".env"
load_dotenv(dotenv_path=env_path)
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_USERNAME = os.getenv("OLLAMA_USERNAME")
OLLAMA_PASSWORD = os.getenv("OLLAMA_PASSWORD")

# To run this API:
# 'uv run fastapi dev api.py'
# Go to /docs/ endpoint to test functionality


def contact_type(contact_detail: str) -> tuple:
    if contact_detail == "social_media":
        return (SocialMedia, ...)
    else:
        return (str, ...)


def create_contact_dict(contact_details: list[str]) -> dict:
    contact_dict = {}
    for c in contact_details:
        contact_dict[c] = contact_type(c)
    return contact_dict


class SocialMedia(BaseModel):
    linkedin: str
    twitter_x: str
    facebook: str
    telegram: str
    signal: str
    instagram: str


async def fetch_web_page(target_url):
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(target_url)


class SystemMessage:
    def __init__(self, occupations: list[str] | None) -> None:
        self.occupations = ", ".join(occupations) if occupations else None
        self.personality = "You are a data extraction assistant specialized in extracting contact information from webpages."
        if self.occupations:
            self.task = f"TASK: Extract contact details of following occupations/roles found in the provided webpage content: ({self.occupations})"
        else:
            self.task = f"TASK: Extract all contact details found in the provided webpage content."
        self.output_format = "OUTPUT FORMAT: Return a valid JSON list of objects."
        self.rules = (
            "RULES:- Construct the email address even if one isn't explicitly provided."
        )

    def form(self):
        sys_message = self.personality + "\n" + self.task + "\n" + self.rules
        return sys_message


class UserPrompt:
    def __init__(self, web_content: str):
        self.prompt = f"Here is the content of a webpage:\n\n{web_content}"

    def form(self):
        return self.prompt


class SeekParameters(BaseModel):
    contact_details: list[str]
    occupations: list[str]
    url: str


app = FastAPI()


@app.post("/seek/")
async def process_request(parameters: SeekParameters):
    result = await fetch_web_page(parameters.url)
    if not result.success:
        raise HTTPException(status_code=404, detail="Unable to fetch web page.")
    markdown_content = result.markdown
    system_message = SystemMessage(parameters.occupations)
    user_prompt = UserPrompt(markdown_content)
    contact_dict = create_contact_dict(parameters.contact_details)
    ContactInfo = create_model("ContactInfo", **contact_dict)

    class ContactList(BaseModel):
        contacts: List[ContactInfo]

    model = ChatOllama(
        base_url=f"https://{OLLAMA_USERNAME}:{OLLAMA_PASSWORD}@{OLLAMA_URL}:{OLLAMA_PORT}",
        model="ministral-3:8b",
        format="json",
        temperature=0.0,
        top_p=0.5,
        num_predict=65536,
        num_ctx=65536,
        # client_kwargs are given straight to httpx client
        client_kwargs={"verify": False},  # because of private SSL cert
    )

    messages = [("system", system_message.form()), ("human", user_prompt.form())]

    structured_model = model.with_structured_output(ContactList)
    result = structured_model.invoke(messages)

    return result

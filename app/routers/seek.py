from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from models.contact import SeekParameters, build_contact_list_model
from services.crawler import fetch_web_page
from services.llm import build_ollama_instance
from prompts.contact_extraction import SystemMessage, UserPrompt, build_messages

router = APIRouter()


@router.post("/seek/")
async def process_request(parameters: SeekParameters):
    result = await fetch_web_page(parameters.url)
    if not result.success:
        raise HTTPException(status_code=404, detail="Unable to fetch web page.")
    markdown_content = result.markdown
    system_message = SystemMessage(parameters.occupations)
    user_prompt = UserPrompt(markdown_content)
    messages = build_messages(system_message=system_message, user_prompt=user_prompt)
    ContactList = build_contact_list_model(parameters.contact_details)
    model: ChatOllama = build_ollama_instance(
        model=parameters.model,
        temp=parameters.temp,
        top_p=parameters.top_p,
        num_predict=parameters.num_predict,
        num_ctx=parameters.num_ctx,
    )
    structured_model = model.with_structured_output(ContactList)
    result = structured_model.invoke(messages)

    return result
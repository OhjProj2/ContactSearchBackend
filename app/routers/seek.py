from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure
from timer import Timer

from models.contact import SeekParameters, build_contact_list_model
from services.crawler import fetch_web_page
from services.llm import build_ollama_instance
from prompts.contact_extraction import SystemMessage, UserPrompt, build_messages
from services import mongodb
import config

settings = config.Settings()
router = APIRouter()


@router.post("/seek/")
async def process_request(parameters: SeekParameters):
    """Router that gets a structured list of contact details from Ollama model

    - Fetches web page content
    - Creates messages to be sent to LLM
    - Builds ContactList class that's used to structure the answer
    - Builds a ChatOllama model instance
    - Gives the model instance a structure to follow
    - Invokes a call to the model
    - Saves the result to MongoDB
    - Returns the result

    Args:
        parameters: SeekParameters containing contact_details, occupations,
        URL + model parameters + MongoDB parameters

    Returns:
        Structured contact list extracted from the page content.
    """
    try:
        await mongodb.ping(host=parameters.db_uri)
    except ConnectionFailure:
        raise HTTPException(status_code=503, detail="Database unavailable")

    async with Timer() as timer:
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

    await mongodb.add_contact_details(
        contact_details=result,
        seek_parameters=parameters,
    )

    return {
        "data": result,
        "time": round(timer.duration, 4)
    }

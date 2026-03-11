import asyncio

from fastapi import HTTPException
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, InvalidURI

from routers import seek
from models import contact
import config

settings = config.Settings()


async def ping(host: str):
    """Pings MongoDB server

    Args:
        host: MongoDB server URI

    Raises:
        HTTPException ConnectionFailure: If server doesn't respond
    """
    client = AsyncMongoClient(host=host, server_api=ServerApi("1"))
    try:
        await client.admin.command("ping")
    except ConnectionFailure:
        raise HTTPException(status_code=500, detail="No connection to database")
    except InvalidURI:
        raise HTTPException(status_code=500, detail="Invalid URI")


async def add_contact_details(
    contact_details,  # fetched contact details structured using dynmically generated ContactList
    seek_parameters: seek.SeekParameters,
):
    """Asynchronous method that sends fetched contact details to MongoDB.

    Args:
        contact_details: a result object created by structured_model.invoke
        seek_parameters: contains contact_details, occupations, URL + model
        parameters + MongoDB parameters

    Raises:
        HTTPException: In case the result is not acknowledged or any exception
    """
    client = AsyncMongoClient(host=seek_parameters.db_uri, server_api=ServerApi("1"))
    db = client[seek_parameters.db_name]
    collection = db[seek_parameters.db_collection]
    query_results = {
        "seek_parameters": seek_parameters.model_dump(),
        "contact_details": contact_details.model_dump(),
    }
    try:
        result = await collection.insert_one(query_results)
        if not result.acknowledged:
            raise HTTPException(
                status_code=500, detail="Write not acknowledged by MongoDB"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Write to database failed: {str(e)}"
        )
    return result


async def get_contact_details(db_name: str, db_collection: str):
    pass


async def get_collections(db_name: str) -> list[str]:
    pass


async def get_dbs() -> list[str]:
    pass


async def test():
    seek_parameters = contact.SeekParameters(
        contact_details=[
            "school",
            "first_name",
            "last_name",
            "email",
            "occupation",
            "role",
            "phone_number",
        ],
        occupations=["Rehtori", "Sihteeri"],
        url="https://www.vihti.fi/kasvatus-ja-koulutus/perusopetus/7-9-luokkien-koulut/nummelanharjun-koulu/henkilokunta/",
    )
    data = await seek.process_request(seek_parameters)
    print(data.model_dump())


if __name__ == "__main__":
    asyncio.run(test())

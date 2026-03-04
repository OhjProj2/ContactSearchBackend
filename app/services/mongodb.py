import asyncio
from re import A

from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure

from routers import seek
from models import contact
import config

settings = config.Settings()


async def ping(client: AsyncMongoClient):
    try:
        await client.admin.command("ping")
    except ConnectionFailure:
        print("Server not available")


async def add_contact_details(
    contact_details,  # dynamically generated ContactList
    seek_parameters: seek.SeekParameters,
):
    """Asynchronous method that sends fetched contact details to MongoDB

    Args:
        contact_details: a result object created by structured_model.invoke
        seek_parameters: contains contact_details, occupations, URL + model
        parameters + MongoDB parameters
    """
    client = AsyncMongoClient(host=seek_parameters.db_uri)
    await ping(client=client)
    db = client[seek_parameters.db_name]
    collection = db[seek_parameters.db_collection]
    query_results = {
        "seek_parameters": seek_parameters.model_dump(),
        "contact_details": contact_details.model_dump(),
    }
    result = await collection.insert_one(query_results)
    print(f"Inserted {result.inserted_id} object.")
    return result.inserted_id


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

import asyncio
from bson import ObjectId

from fastapi import HTTPException
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, InvalidURI

from app.routers import seek
from app.models import contact
from app.config import Settings

settings = Settings()

client = AsyncMongoClient(host=settings.MONGODB_URI, server_api=ServerApi("1"))


async def ping(host: str):
    """Pings MongoDB server

    Args:
        host: MongoDB server URI

    Raises:
        HTTPException ConnectionFailure: If server doesn't respond
    """
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


async def get_all_data(db_name: str, db_collection: str):
    """Asynchronous method that fetches all data from specific collection.

    Args:
        db_name: name of the database
        db_collection: name of the collection in given database

    Raises:
        HTTPException: In case of any error
    Return:
        All data of single collection.
    """
    db = client[db_name]
    collection = db[db_collection]
    try:
        cursor = collection.find({})
        result = await cursor.to_list(length=None)
        for doc in result:
            doc["_id"] = str(doc["_id"])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Read from database failed: {str(e)}"
        )
    return result


async def _get_data_by_id(
    id: str,
    db_name: str = settings.MONGODB_NAME,
    db_collection: str = settings.MONGODB_COLLECTION,
):
    """Asynchronous method that fetches data of document in specific database+collection by id number.

    Args:
        id: id number of single document
        db_name: name of the database (uses default database if not given)
        db_collection: name of the collection in given database (uses default database if not given)

    Raises:
        HTTPException: In case of any error

    Returns:
        All data of single document.

    """
    db = client[db_name]
    collection = db[db_collection]
    try:
        result = await collection.find_one({"_id": ObjectId(id)})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Read from database failed: {str(e)}"
        )
    if result:
        result["_id"] = str(result["_id"])
    return result


async def get_collections(db_name: str) -> list[str]:
    """Asynchronous method that fetches all collections in specific database.

    Args:
        db_name: name of the database

    Raises:
        HTTPException: In case of any error
    """
    db = client[db_name]
    try:
        result = await db.list_collection_names()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Read from database failed: {str(e)}"
        )
    return result


async def get_dbs() -> list[str]:
    """Asynchronous method that fetches all databases not on ignore list.

    Args:
        None
    Raises:
        HTTPException: In case of any error
    """
    ignored_dbs = ["admin", "local"]
    try:
        all_dbs = await client.list_database_names()
        result = [db for db in all_dbs if db not in ignored_dbs]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Read from database failed: {str(e)}"
        )

    return result


async def copy_id_to(id: str, to_database: str, to_collection: str):
    """Copies one document by id from the default source collection to another collection.

    Args:
        id: MongoDB document id from the default database and collection.
        to_database: name of the target database
        to_collection: name of the target collection in the target database

    Returns:
        Dictionary containing the inserted document id in the target collection.

    Raises:
        HTTPException 404: if the source document is not found.
        HTTPException 500: if reading from the source or saving to the target fails.
    """
    db = client[to_database]
    collection = db[to_collection]
    try:
        data = await _get_data_by_id(id=id)
        if not data:
            raise HTTPException(status_code=404, detail=f"Document {id} not found")

        data.pop("_id", None)
        result = await collection.insert_one(data)
        if not result.acknowledged:
            raise HTTPException(
                status_code=500,
                detail=f"Save to {to_database}/{to_collection} was not acknowledged",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Save to {to_database}/{to_collection} failed: {str(e)}",
        )

    return {"inserted_id": str(result.inserted_id)}


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
    # asyncio.run(test())
    # asyncio.run(get_collections("testdatabase"))
    # asyncio.run(get_dbs())
    # asyncio.run(get_contact_details("testdatabase", "testcollection"))
    asyncio.run(copy_id_to("69a9a24f88de37b2a85b72b6", "fuk", "phuk"))

import json

from fastapi import APIRouter, HTTPException
from app.services import mongodb
from app.config import Settings

settings = Settings()

router = APIRouter()


@router.get("/listalldata")
async def listalldata(
    db_name: str = settings.MONGODB_NAME,
    db_collection: str = settings.MONGODB_COLLECTION,
):
    """Router that gets list of all data in one collection on MongoDB server."""
    try:
        response = await mongodb.get_all_data(db_name, db_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

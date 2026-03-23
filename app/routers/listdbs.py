import json

from fastapi import APIRouter, HTTPException
from app.services import mongodb


router = APIRouter()


@router.get("/listdbs")
async def list_dbs():
    """Router that gets list of all databases containing fetched data on MongoDB server."""
    try:
        response = await mongodb.get_dbs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

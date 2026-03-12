import json

from fastapi import APIRouter, HTTPException
from services import mongodb

    
router = APIRouter()
@router.post("/listcontactdetails")
async def list_contact_details(db_name: str, db_collection: str):
    """Router that gets list of all data in one collection on MongoDB server."""
    try:
        response = await mongodb.get_contact_details(db_name, db_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

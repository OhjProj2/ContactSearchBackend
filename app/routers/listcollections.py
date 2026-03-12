import json

from fastapi import APIRouter, HTTPException
from services import mongodb

    
router = APIRouter()
@router.post("/listcollections")
async def list_collections(db_name: str):
    """Router that gets list of all colletions in one database on MongoDB server."""
    try:
        response = await mongodb.get_collections(db_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response 

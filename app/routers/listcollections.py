from fastapi import APIRouter, HTTPException
from services import mongodb
from config import Settings

settings = Settings()
    
router = APIRouter()
@router.post("/listcollections")
async def list_collections(db_name: str = settings.MONGODB_NAME):
    """Router that gets list of all colletions in one database on MongoDB server."""
    try:
        response = await mongodb.get_collections(db_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response 

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import mongodb
from app.config import Settings

settings = Settings()

router = APIRouter()


class CopyRequest(BaseModel):
    id: str
    db_name: str = settings.MONGODB_NAME
    col_name: str = settings.MONGODB_COLLECTION


@router.post("/copybyid")
async def copybyid(request: CopyRequest):
    try:
        response = await mongodb.copy_id_to(
            id=request.id,
            to_database=request.db_name,
            to_collection=request.col_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

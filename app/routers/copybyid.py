from fastapi import APIRouter, HTTPException
from app.services import mongodb
from app.config import Settings

settings = Settings()

router = APIRouter()


@router.post("/copybyid")
async def copybyid(
    id: str,
    db_name: str = settings.MONGODB_NAME,
    col_name: str = settings.MONGODB_COLLECTION,
):
    """Router that copies data on specific id in default database+collection
    to a given database+collection.
    """
    try:
        response = await mongodb.copy_id_to(
            id=id,
            to_database=db_name,
            to_collection=col_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response

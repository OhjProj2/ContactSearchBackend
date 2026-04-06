from fastapi import APIRouter, HTTPException
from app.models.contact import default_fields

router = APIRouter()


@router.get("/listfields")
async def list_fields():
    """Router that gets list of all default fields."""
    return [{"label": field.name, "value": field.var_name} for field in default_fields]
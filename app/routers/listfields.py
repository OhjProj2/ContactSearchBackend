from fastapi import APIRouter, HTTPException
from app.models.contact import default_fields

router = APIRouter()


@router.get("/listfields")
async def list_fields():
    """Router that gets list of all default fields."""
    default_field_names = [ContactField.name for ContactField in default_fields]
    return default_field_names

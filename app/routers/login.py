from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import Settings

router = APIRouter()
settings = Settings()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(data: LoginRequest):
    if data.username == settings.ADMIN_USERNAME and data.password == settings.ADMIN_PASSWORD:
        return {"success": True}

    raise HTTPException(status_code=401, detail="Invalid credentials")
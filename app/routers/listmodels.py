from fastapi import APIRouter, HTTPException
from config import Settings
import requests
from requests.auth import HTTPBasicAuth

settings = Settings()
router = APIRouter()
basic = HTTPBasicAuth(settings.OLLAMA_USERNAME, settings.OLLAMA_PASSWORD)

@router.post("/listmodels")
async def list_models():
    try:
        response = requests.get(f"https://{settings.OLLAMA_URL}:{settings.OLLAMA_PORT}/api/tags", auth=basic)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=str(e))    
from fastapi import APIRouter, HTTPException
from config import Settings
import httpx

settings = Settings()
router = APIRouter()


@router.post("/listmodels")
async def list_models():
    """Router that gets list of model details from Ollama server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"https://{settings.OLLAMA_URL}:{settings.OLLAMA_PORT}/api/tags",
                auth=(settings.OLLAMA_USERNAME, settings.OLLAMA_PASSWORD),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


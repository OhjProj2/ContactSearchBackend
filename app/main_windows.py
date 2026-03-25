#run with: uv run python main_windows.py

import asyncio
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from app.routers.seek import router as seek_router
from app.routers.listmodels import router as listmodels_router
from app.routers.listdbs import router as listdbs_router
from app.routers.listcollections import router as listcollections_router
from app.routers.listalldata import router as listalldata_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(seek_router, tags=["Contact Extraction"])
app.include_router(listmodels_router, tags=["List Models"])
app.include_router(listdbs_router, tags=["List Databases"])
app.include_router(listcollections_router, tags=["List Collections"])
app.include_router(listalldata_router, tags=["List Saved Data"])

if __name__ == "__main__":
    uvicorn.run("app.main_windows:app", host="127.0.0.1", port=8000, reload=False)
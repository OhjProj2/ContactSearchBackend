from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.seek import router as seek_router
from app.routers.listmodels import router as listmodels_router

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(seek_router, tags=["Contact Extraction"])
app.include_router(listmodels_router, tags=["List Models"])

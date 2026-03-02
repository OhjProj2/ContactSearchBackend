from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.seek import router as seek_router
from routers.listmodels import router as listmodels_router
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

from routers.seek import router as seek_router

app.include_router(seek_router, tags=["Contact Extraction"])

from fastapi import FastAPI

from routers.seek import router as seek_router

app = FastAPI()
app.include_router(seek_router, tags=["Contact Extraction"])


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.seek import router as seek_router
from app.routers.listmodels import router as listmodels_router
from app.routers.listdbs import router as listdbs_router
from app.routers.listcollections import router as listcollections_router
from app.routers.listalldata import router as listalldata_router
from app.routers.listallcontactdata import router as listallcontactdata_router
from app.routers.login import router as login_router
from app.routers.listfields import router as listfields_router
from app.routers.copybyid import router as copybyid_router

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
app.include_router(listdbs_router, tags=["List Databases"])
app.include_router(listcollections_router, tags=["List Collections"])
app.include_router(listalldata_router, tags=["List All Saved Data"])
app.include_router(listallcontactdata_router, tags=["List Saved Contact Data"])
app.include_router(login_router, tags=["Authentication"])
app.include_router(listfields_router, tags=["List Default Contact Detail Fields"])
app.include_router(copybyid_router, tags=["Copy document to db+collection"])

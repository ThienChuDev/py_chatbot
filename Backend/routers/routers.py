from fastapi import FastAPI
from routers.homeRouter import router as homeRouter


def routers(app: FastAPI):
    app.include_router(homeRouter, prefix="/home", tags=["Home"])


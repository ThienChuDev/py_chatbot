from fastapi import APIRouter
from controllers.HomeController import HomeController
router = APIRouter()
from pydantic import BaseModel

class Item(BaseModel):
    input: str

@router.get("/")
def home():
    return HomeController.test()

@router.post("/input")
def home(item: Item):
    return HomeController.input(item)

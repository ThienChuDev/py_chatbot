from typing import Union
from fastapi import FastAPI
import uvicorn
import asyncio
from routers import routers
from config.firebase import conected_firebase
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app = FastAPI()

origins = [
    "http://10.100.101.0:3000",  
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


routers.routers(app)

@app.on_event("startup")
async def on_startup():
    await conected_firebase()
    print("Firebase connected successfully!")

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

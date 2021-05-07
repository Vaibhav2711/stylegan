from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from pydantic import BaseModel

from smile import detect
from pretrained_example import genFace

#from pyngrok import ngrok

#html_url = ngrok.connect(8001,'http',hostname = 'buildar1.ngrok.io')
#print(html_url)

app = FastAPI()

origins = [
    "http://sachin1234.localhost:3000",
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
# pydantic models

class Imageinput(BaseModel):
    seed_num: str

class Imageoutput(BaseModel):
  Smile: str
  Pose: str
# routes

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}


@app.post("/image", response_model=Imageoutput, status_code=200)
async def get_prediction(payload: Imageinput):
    seed_num = payload.seed_num
    image = genFace(seed_num)
    smile,pose = detect(image)
    response_obj = {"Smile":smile,"Pose":pose}
    return response_obj

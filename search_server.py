from typing import Union

from fastapi import FastAPI, UploadFile, File
from search import search, calculate_metrics, pdf_to_pptx
from img_search import search as image_search
from compress import compress
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
import string
app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

def random_slug():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))


class Request(BaseModel):
    body: str


class NumericRequest(BaseModel):
    description: str
    category: str
    type: str

@app.post("/search")
def read_root(body: Request):
    return search(body.body)


@app.post('/numeric')
def get_numeric(body: NumericRequest):
    return calculate_metrics(body.category, body.description, body.type)


@app.post('/compress')
def get_compressed(body: Request):
    return compress(body.body, threshold=0.8)

@app.post('/convert-to-pptx')
async def convert(in_file: UploadFile):
    sl = random_slug()
    with open(f'./static/{sl}.pdf', 'wb') as file:
        content = await in_file.read()
        file.write(content)
        pdf_to_pptx(f'./static/{sl}.pdf')
    return {
        'file': f'/static/{sl}.pptx'
    }

@app.post('/img-search')
async def img_search(body: Request):
    return image_search(body.body)
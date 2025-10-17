from fastapi import FastAPI, HTTPException, UploadFile, File
from src.utils.prediction import get_prediction
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model('model/densenet121_tb.h5')

# Routes
@app.get('/')
def read_root():
    return {"message": "Selamat Datang"}

@app.post("/predict")
async def upload_image(file: UploadFile = File( ... )):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code = 400, detail = "File must be an image")
    
    contents = await file.read()
    result = get_prediction(contents, model)

    return result

# UploadFile is a class from FastAPI that wraps the uploaded files
# It contains use methods such
# file.filename, file.content_type, await file.read(), and file.file()

# File(...) is a dependency inject from FastAPI that tells
# "Expect this parameter to come from a file upload in a multipart/form-data request"

# FastAPI tutorial: https://www.youtube.com/watch?v=iWS9ogMPOI0

# JUST FOR REMEMBERING:
# from pydantic import BaseModel

# class Item(BaseModel):
#     text: str = None
#     is_done: bool = False

# items: list[Item] = []

# @app.post("/items", response_model = Item)
# def create_item(item: Item):
#     items.append(item)
#     return item

# @app.get("/items", response_model = list[Item])
# def list_items():
#     return items

# @app.get("/items/{item_id}", response_model = Item)
# def get_item(item_id: int):
#     if 0 <= item_id < len(items):
#         return items[item_id]
#     else:
#         raise HTTPException(status_code = 404, detail = "Item not found")
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.utils.prediction import get_prediction
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model('model/densenet121_tb.h5')

@app.post("/predict")
async def upload_image(
    file: UploadFile = File( ... ),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code = 400, detail = "File must be an image")
    
    contents = await file.read()
    result = get_prediction(contents, model)

    return result
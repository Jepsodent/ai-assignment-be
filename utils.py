import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
# import os

def get_prediction(contents: bytes):
    # Load model
    model = tf.keras.models.load_model('model/densenet121_tb.h5')

    # Load and preprocess image
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    # Preprocess image
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Interpret result
    if prediction >= 0.5:
        return {"status": "TB", "confidence": float(prediction)}
    else:
        return {"status": "NORMAL", "confidence": float(1 - prediction)}
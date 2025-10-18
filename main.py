import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load model
model = tf.keras.models.load_model('model/densenet121_tb.h5')

# Load and preprocess image
img_path = os.path.join(os.getcwd(), 'test.jpg')
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Interpret result
if prediction >= 0.5:
    print(f"Prediction: TB ({prediction:.2f})")
else:
    print(f"Prediction: NORMAL ({1 - prediction:.2f})")

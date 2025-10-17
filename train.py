import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks

# Configuration
MODEL_PATH = 'model/densenet121_tb.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data Loading
train_gen = ImageDataGenerator(rescale = 1./255,
    zoom_range = 0.2,
    width_shift_range = 0.02,
    height_shift_range =0.02,)
val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_ds = val_gen.flow_from_directory(
    'data/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Model Setup
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Compiling
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop])

# Save Model
os.makedirs('model', exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
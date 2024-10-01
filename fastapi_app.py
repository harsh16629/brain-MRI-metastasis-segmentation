from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import cv2
import nest_asyncio

app = FastAPI()

# Load the trained model
model = load_model('best_model.h5', custom_objects={'dice_coefficient': dice_coefficient})

# Define FastAPI route for image segmentation
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=[0, -1])
    
    # Predict segmentation mask
    prediction = model.predict(image)
    return {"prediction": prediction.tolist()}


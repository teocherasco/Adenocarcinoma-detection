from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/2")

CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# Takes the bytes as an input and return an ndarray as an output
def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    resized = np.expand_dims(image / 255, 0)
    res = tf.image.resize(resized, (224, 224))
    prediction = MODEL.predict(res)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8004)

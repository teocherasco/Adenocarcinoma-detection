from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

BUCKET_NAME = "mateocherasco-tf-models"
CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

model = None


# Download the model, BLOB (Binary Large Object)
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/adenocarcinoma_detection.h5",
            "/tmp/adenocarcinoma_detection.h5"
        )

        model = tf.keras.models.load_model("/tmp/adenocarcinoma_detection.h5")

    image = request.files["file"]

    image = np.array(Image.open(image).convert("RGB"))
    resized = np.expand_dims(image / 255, 0)
    res = tf.image.resize(resized, (224, 224))

    pred = model.predict(res)
    print(pred)

    predicted_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = round(100 * (np.max(pred[0])), 2)

    return {"Class": predicted_class, "Confidence": confidence}



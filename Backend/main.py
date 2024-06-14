from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the feature extractor layer for loading the model
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
KerasLayer = hub.KerasLayer

# Load the trained model
model = load_model('sign_language_model.h5', custom_objects={'KerasLayer': KerasLayer})

# Define class indices for the entire alphabet
class_indices = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}
inv_class_indices = {v: k for k, v in class_indices.items()}

def preprocess_image(image: Image.Image, img_size=(224, 224)):
    img = image.resize(img_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes))

    img = preprocess_image(image)
    
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    sign_name = inv_class_indices[class_idx]

    return JSONResponse(content={"letter": sign_name})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

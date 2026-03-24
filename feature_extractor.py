import os
import numpy as np
from PIL import Image
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

DATASET_PATH = "static/dataset"
IMAGE_SIZE = (224, 224)

# 🔍 DEBUG LINES (ADD HERE)
print("Current Working Directory:", os.getcwd())
print("Dataset Path Exists:", os.path.exists(DATASET_PATH))
print("Files in Dataset:", os.listdir(DATASET_PATH))

features = []
image_names = []

for img_name in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = model.predict(img_array, verbose=0)
        features.append(feature[0])
        image_names.append(img_name)

        print(f"Processed: {img_name}")

    except Exception as e:
        print(f"Skipping {img_name}: {e}")

np.save("features.npy", np.array(features))
np.save("image_names.npy", np.array(image_names))


print("✅ Feature extraction complete!")
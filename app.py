import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DATASET_FOLDER = "static/dataset/images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

features = np.load("features.npy")
image_names = np.load("image_names.npy")

def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = model.predict(img_array, verbose=0)
    return feature.flatten()   # ✅ FIXED

@app.route("/", methods=["GET", "POST"])
def index():
    query_image = None
    similar_images = []

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = str(uuid.uuid4()) + "_" + file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            query_image = filename

            query_feature = extract_feature(save_path)

            similarity = cosine_similarity([query_feature], features)[0]  # ✅ FIXED

            top_k = similarity.argsort()[-5:][::-1]

            for idx in top_k:
                similar_images.append({
                    "name": image_names[idx],
                    "score": round(float(similarity[idx]), 3)
                })

    return render_template("index.html", query_image=query_image, similar_images=similar_images)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Configuration des uploads
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'  # Dossier où enregistrer les images
configure_uploads(app, photos)

# Charger le modèle ResNet50 pré-entraîné
model = ResNet50(weights="imagenet")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensionnement
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    img_array = preprocess_input(img_array)  # Normalisation
    return img_array

@app.route("/",)
def home():
    return render_template("index.html")

@app.route("/predictImage", methods=["POST"])
def predictImage():
    if "photo" in request.files:
        filename = photos.save(request.files["photo"])
        img_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename)

        # Prétraiter et prédire la classe
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        return render_template("result.html", filename=filename, predictions=decoded_predictions)

    return render_template("index.html", error="Aucune image sélectionnée !")

if __name__ == '__main__':
    app.run(debug=True)

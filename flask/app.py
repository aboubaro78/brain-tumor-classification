import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json

try:
    import tensorflow as tf
    print("TensorFlow importé avec succès. Version :", tf.__version__)
except ImportError as e:
    tf = None
    print("Erreur lors de l'importation de TensorFlow :", e)

app = Flask(__name__)

# Charger le modèle PyTorch
try:
    from models.cnn import CNN1
    pytorch_model = CNN1()
    pytorch_model.load_state_dict(torch.load("models/Abou_Birane_model.torch", map_location=torch.device('cpu'), weights_only=True))
    pytorch_model.eval()
    print("Modèle PyTorch chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle PyTorch : {e}")
    pytorch_model = None

# Charger le modèle TensorFlow
tensorflow_model = None
if tf is not None:
    model_path = "models/Abou_Birane_model.tensorflow"
    print(f"Tentative de chargement du modèle TensorFlow depuis : {os.path.abspath(model_path)}")
    if not os.path.exists(model_path):
        print(f"Erreur : Le chemin {model_path} n'existe pas.")
    else:
        try:
            tensorflow_model = tf.keras.models.load_model(model_path)
            print("Modèle TensorFlow chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle TensorFlow : {e}")
else:
    print("TensorFlow non disponible (tf est None).")

# Charger les métriques
def load_metrics(model_type):
    metrics_path = f"models/Abou_Birane_{model_type}_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

# Transformation pour PyTorch
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    probabilities = None
    selected_model = request.form.get('model') if request.method == 'POST' else None
    metrics = None

    # Charger les métriques après sélection du modèle
    if selected_model:
        metrics = load_metrics(selected_model)

    if request.method == 'POST':
        try:
            model_type = request.form['model']
            uploaded_file = request.files.get('image_path')

            print(f"Modèle sélectionné : {model_type}, Fichier uploadé : {uploaded_file.filename if uploaded_file else 'Aucun'}")

            if not uploaded_file or uploaded_file.filename == '':
                error = "Aucune image téléchargée. Veuillez sélectionner un fichier."
                print("Erreur : Aucune image téléchargée.")
            else:
                print("Ouverture de l'image uploadée...")
                img = Image.open(uploaded_file).convert('RGB')
                print("Image ouverte avec succès.")

                if model_type == 'pytorch':
                    print("Traitement avec PyTorch...")
                    if pytorch_model is None:
                        error = "Modèle PyTorch non disponible."
                        print("Erreur : Modèle PyTorch non disponible.")
                    else:
                        img_tensor = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            output = pytorch_model(img_tensor)
                            probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy().tolist()
                            pred = torch.argmax(output, dim=1).item()
                        prediction = classes[pred]
                        print(f"Prédiction PyTorch : {prediction}, Probabilités : {probabilities}")
                elif model_type == 'tensorflow':
                    print("Traitement avec TensorFlow...")
                    if tensorflow_model is None or tf is None:
                        error = "Modèle TensorFlow non disponible."
                        print("Erreur : Modèle TensorFlow non chargé ou tf est None.")
                    else:
                        print("Préparation de l'image pour TensorFlow...")
                        img_array = np.array(img.resize((224, 224))) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        print("Dimensions de l'image après expand_dims :", img_array.shape)
                        print("Exécution de la prédiction TensorFlow...")
                        output = tensorflow_model.predict(img_array, verbose=0)
                        probabilities = output[0].tolist()
                        pred = np.argmax(output, axis=1)[0]
                        prediction = classes[pred]
                        print(f"Prédiction TensorFlow : {prediction}, Probabilités : {probabilities}")
                else:
                    error = "Modèle non valide."
                    print("Erreur : Modèle non valide.")
        except Exception as e:
            error = f"Erreur lors de la prédiction : {str(e)}"
            print(f"Erreur lors de la prédiction : {str(e)}")

    return render_template('index.html', prediction=prediction, error=error, 
                          selected_model=selected_model, metrics=metrics, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
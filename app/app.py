from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import folium
import json
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Charger ton modèle YOLOv8
model = YOLO('../model/best.pt')

# Fonction pour charger les entrées depuis le fichier entries.json
def load_entries():
    try:
        with open('entries.json', 'r') as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    return entries

# Fonction pour sauvegarder les entrées dans entries.json
def save_entries(entries):
    with open('entries.json', 'w') as f:
        json.dump(entries, f)

# Fonction pour ajouter une entrée dans entries.json
def add_entry(entry):
    entries = load_entries()  # Charger les anciennes entrées

    # Chercher si une entrée avec les mêmes coordonnées existe déjà
    found = False
    for existing_entry in entries:
        if existing_entry['latitude'] == entry['latitude'] and existing_entry['longitude'] == entry['longitude']:
            existing_entry['object_classes'][entry['object_class']] = existing_entry['object_classes'].get(entry['object_class'], 0) + 1
            found = True
            break

    if not found:
        entry['object_classes'] = {entry['object_class']: 1}
        entries.append(entry)

    save_entries(entries)  # Sauvegarder les nouvelles entrées dans le fichier

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Lire l'image
    img = Image.open(file.stream)

    # Vérifier si l'image a un canal alpha (transparence)
    if img.mode == 'RGBA':  # RGBA contient un canal alpha
        img = img.convert('RGB')  # Convertir en RGB (supprimer le canal alpha)
    
    img = np.array(img)

    # Effectuer les prédictions avec YOLOv8
    results = model(img)
    result = results[0]  # Garder uniquement les prédictions de la première image
    
    boxes = result.boxes

    # Déplacer les tenseurs du GPU vers le CPU avant de les convertir en numpy
    xywh_cpu = boxes.xywh.cpu().numpy()
    conf_cpu = boxes.conf.cpu().numpy()
    cls_cpu = boxes.cls.cpu().numpy()

    # Accéder aux noms de classes depuis 'result.names'
    class_names = result.names  # Cela retourne un dictionnaire de classe ID -> nom de la classe

    # Créer un DataFrame avec les données
    df = pd.DataFrame({
        'x': xywh_cpu[:, 0],  # Coordonnée x
        'y': xywh_cpu[:, 1],  # Coordonnée y
        'width': xywh_cpu[:, 2],  # Largeur de la boîte
        'height': xywh_cpu[:, 3],  # Hauteur de la boîte
        'confidence': conf_cpu,  # Confiance
        'class': cls_cpu  # Classes (identifiant de classe)
    })

    # Ajouter les noms des classes
    df['name'] = df['class'].apply(lambda x: class_names.get(int(x), 'Unknown'))  # Appliquer le nom de la classe à chaque ligne

    # Filtrer les détections par confiance
    detections = df[df['confidence'] > 0.5]  # Seulement les détections avec une confiance > 0.5

    # Supposons que l'utilisateur fournisse aussi les coordonnées
    lat = float(request.form.get('latitude', 0))
    lon = float(request.form.get('longitude', 0))

    if lat == 0 or lon == 0:
        return 'Coordinates not provided or invalid', 400

    # Ajouter les nouvelles entrées dans le fichier entries.json
    for _, detection in detections.iterrows():
        entry = {
            'latitude': lat,
            'longitude': lon,
            'object_class': detection['name']
        }
        add_entry(entry)

    # Créer une carte Folium centrée sur la position
    map_ = folium.Map(location=[lat, lon], zoom_start=12)

    # Charger toutes les entrées depuis le fichier pour les afficher sur la carte
    entries = load_entries()

    # Regrouper les entrées par coordonnées (latitude, longitude)
    grouped_entries = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        grouped_entries[(entry['latitude'], entry['longitude'])].update(entry['object_classes'])

    # Ajouter des marqueurs sur la carte
    for (lat, lon), object_classes in grouped_entries.items():
        label = ', '.join([f"{cls}: {count}" for cls, count in object_classes.items()])
        icon = folium.Icon(icon='cloud', icon_size=(35, 35), color='blue')
        folium.Marker([lat, lon], popup=label, icon=icon).add_to(map_)

    # Sauvegarder la carte dans un fichier HTML temporaire
    map_html = 'static/map.html'
    map_.save(map_html)

    return render_template('result.html', map_html=map_html, latitude=lat, longitude=lon)

if __name__ == '__main__':
    app.run(debug=True)

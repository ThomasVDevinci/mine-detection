from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import folium
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import pandas as pd
from folium import plugins

app = Flask(__name__)

# Charger ton modèle YOLOv8
model = YOLO('../model/best.pt')

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

    # Supposons que l'utilisateur fournisse aussi les métadonnées de localisation
    # Vérification des coordonnées
    lat = float(request.form.get('latitude', 0))
    lon = float(request.form.get('longitude', 0))

    if lat == 0 or lon == 0:
        return 'Coordinates not provided or invalid', 400

    # Créer une carte Folium centrée sur la position
    map_ = folium.Map(location=[lat, lon], zoom_start=12)

    # Ajouter un marqueur pour chaque objet détecté
    for _, detection in detections.iterrows():
        label = detection['name']
        x_center, y_center = detection['x'], detection['y']
        # Conversion simplifiée des coordonnées x et y en coordonnées GPS
        marker_lat = lat + (y_center - 240) * 0.0001  # Calculer un ajustement simplifié pour la latitude
        marker_lon = lon + (x_center - 320) * 0.0001  # Calculer un ajustement simplifié pour la longitude

        # Créer un icône personnalisé pour le marqueur avec une taille agrandie
        icon = folium.Icon(icon='cloud', icon_size=(35, 35), color='blue')

        # Ajouter le marqueur avec l'info de classe dans l'infobulle
        folium.Marker(
            [marker_lat, marker_lon], 
            popup=f"Class: {label}", 
            icon=icon
        ).add_to(map_)

    # Sauvegarder la carte dans un fichier HTML temporaire
    map_html = 'static/map.html'
    map_.save(map_html)

    # Convertir l'image en base64 pour l'afficher dans la page HTML
    buffered = BytesIO()
    img = Image.fromarray(img)
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return render_template('result.html', map_html=map_html, latitude=lat, longitude=lon, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)

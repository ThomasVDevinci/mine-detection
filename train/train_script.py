from ultralytics import YOLO

if __name__ == "__main__":
    # Charger un modèle pré-entraîné
    model = YOLO('yolov8n.pt')

    # Entraînement du modèle avec les paramètres corrigés
    model.train(
    data="../data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda',
    workers=2  # Réduire le nombre de workers
)


from ultralytics import YOLO
model = YOLO("best.pt")
model.save("best_clean.pt")
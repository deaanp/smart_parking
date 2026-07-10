from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="./datasets/brd-3/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    # device=0
)      
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/Users/minsolkang/CV/datasets/food/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.005,
    optimizer="AdamW",
    weight_decay=0.0005,
    device="mps"   #"cpu"
)
from ultralytics import YOLO

model = YOLO("/Users/minsolkang/CV/runs/detect/train3/weights/best.pt")
metrics = model.val(data="/Users/minsolkang/CV/datasets/food/data.yaml")
print(metrics.box.map)     # mAP50-95
print(metrics.box.map50)   # mAP50
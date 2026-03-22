from ultralytics import YOLO

model = YOLO("weights/yolov8/yolov8m.pt")
metrics = model.val(
    data="datasets/coco/coco8.yaml",
    imgsz=640,
    batch=32,
    conf=0.001,
    iou=0.7,
    device="cpu",
    save_json=True
)

print(metrics.box.map, metrics.box.map50, metrics.box.map75)


'''
curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
curl -L -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
'''
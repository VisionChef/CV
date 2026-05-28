from ultralytics import YOLO

# 학습된 best.pt 불러오기
model = YOLO("/Users/minsolkang/CV/runs/detect/train2/weights/best.pt")

# 한 장 이미지 추론
results = model.predict(
    source="/Users/minsolkang/test_images",
    conf=0.25,
    save=True,
    device="mps"   # CPU면 "cpu"
)

print(results)
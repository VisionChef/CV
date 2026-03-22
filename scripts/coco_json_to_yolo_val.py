import json
import os
from collections import defaultdict

# 경로 설정
BASE_DIR = "/Users/minsolkang/CV/datasets/coco"
JSON_PATH = os.path.join(BASE_DIR, "annotations", "instances_val2017.json")
IMAGE_DIR = os.path.join(BASE_DIR, "images", "val2017")
LABEL_DIR = os.path.join(BASE_DIR, "labels", "val2017")

os.makedirs(LABEL_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    coco = json.load(f)

# COCO category_id -> YOLO class_id(0~79) 매핑
categories = coco["categories"]
cat_id_to_yolo = {}
for i, cat in enumerate(sorted(categories, key=lambda x: x["id"])):
    cat_id_to_yolo[cat["id"]] = i

# image_id -> file_name, width, height
images = {}
for img in coco["images"]:
    images[img["id"]] = {
        "file_name": img["file_name"],
        "width": img["width"],
        "height": img["height"],
    }

# 각 이미지별 annotation 모으기
image_to_annotations = defaultdict(list)
for ann in coco["annotations"]:
    if ann.get("iscrowd", 0) == 1:
        continue  # 보통 YOLO 학습/평가용 txt 변환에서는 crowd 제외
    image_to_annotations[ann["image_id"]].append(ann)

# 이미지별 txt 생성
for image_id, info in images.items():
    file_name = info["file_name"]
    width = info["width"]
    height = info["height"]

    txt_name = os.path.splitext(file_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    anns = image_to_annotations.get(image_id, [])

    with open(txt_path, "w") as f:
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO: top-left x,y,width,height
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            class_id = cat_id_to_yolo[ann["category_id"]]

            # 값 범위 안전 보정
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("변환 완료")
print(f"JSON: {JSON_PATH}")
print(f"IMAGE_DIR: {IMAGE_DIR}")
print(f"LABEL_DIR: {LABEL_DIR}")
print(f"총 이미지 수: {len(images)}")
print(f"라벨 txt 생성 완료")
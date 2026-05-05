import json
from pathlib import Path
from collections import Counter, defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(data_yaml: Path) -> dict:
    try:
        import yaml

        if not data_yaml.exists():
            return {}

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

        names = data.get("names", {})
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}

        return {}

    except Exception:
        return {}


def find_images(dataset_dir: Path):
    return [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def find_labels(dataset_dir: Path):
    return [
        p for p in dataset_dir.rglob("*.txt")
        if "labels" in p.parts
    ]


def image_to_label_path(image_path: Path, dataset_dir: Path) -> Path:
    relative = image_path.relative_to(dataset_dir)
    parts = list(relative.parts)

    if "images" in parts:
        parts[parts.index("images")] = "labels"

    return dataset_dir / Path(*parts).with_suffix(".txt")


def label_to_image_candidates(label_path: Path, dataset_dir: Path):
    relative = label_path.relative_to(dataset_dir)
    parts = list(relative.parts)

    if "labels" in parts:
        parts[parts.index("labels")] = "images"

    base = dataset_dir / Path(*parts).with_suffix("")
    return [base.with_suffix(ext) for ext in IMAGE_EXTS]


def validate_label_file(label_path: Path):
    errors = []
    class_counter = Counter()
    bbox_count = 0

    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append({
                "file": str(label_path),
                "line": line_no,
                "error": "YOLO label must have 5 values: class x_center y_center width height",
            })
            continue

        try:
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                errors.append({
                    "file": str(label_path),
                    "line": line_no,
                    "error": "bbox values out of range",
                    "values": parts,
                })

            class_counter[class_id] += 1
            bbox_count += 1

        except ValueError:
            errors.append({
                "file": str(label_path),
                "line": line_no,
                "error": "invalid numeric value",
                "values": parts,
            })

    return bbox_count, class_counter, errors


def analyze_yolo_dataset(dataset_dir: Path, output_path: Path):
    dataset_dir = dataset_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_yaml = dataset_dir / "data.yaml"
    class_names = load_class_names(data_yaml)

    images = find_images(dataset_dir)
    labels = find_labels(dataset_dir)

    missing_labels = []
    for image in images:
        label_path = image_to_label_path(image, dataset_dir)
        if not label_path.exists():
            missing_labels.append(str(image))

    labels_without_images = []
    for label in labels:
        candidates = label_to_image_candidates(label, dataset_dir)
        if not any(p.exists() for p in candidates):
            labels_without_images.append(str(label))

    class_counter = Counter()
    split_counter = defaultdict(int)
    total_bbox = 0
    label_errors = []

    for label in labels:
        bbox_count, file_class_counter, errors = validate_label_file(label)

        total_bbox += bbox_count
        class_counter.update(file_class_counter)
        label_errors.extend(errors)

        if "train" in label.parts:
            split_counter["train"] += 1
        elif "val" in label.parts:
            split_counter["val"] += 1
        elif "test" in label.parts:
            split_counter["test"] += 1
        else:
            split_counter["unknown"] += 1

    class_distribution = {
        str(class_id): {
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "bbox_count": count,
        }
        for class_id, count in sorted(class_counter.items())
    }

    health_score = 100
    health_score -= min(len(missing_labels), 30)
    health_score -= min(len(labels_without_images), 30)
    health_score -= min(len(label_errors), 40)
    health_score = max(0, health_score)

    report = {
        "dataset_path": str(dataset_dir),
        "summary": {
            "total_images": len(images),
            "total_labels": len(labels),
            "total_bbox": total_bbox,
            "total_classes": len(class_counter),
            "health_score": health_score,
        },
        "class_distribution": class_distribution,
        "split_label_count": dict(split_counter),
        "issues": {
            "missing_labels": missing_labels,
            "labels_without_images": labels_without_images,
            "label_errors": label_errors,
        },
        "recommendations": build_recommendations(
            class_counter=class_counter,
            missing_labels=missing_labels,
            labels_without_images=labels_without_images,
            label_errors=label_errors,
        ),
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report


def build_recommendations(
    class_counter: Counter,
    missing_labels: list,
    labels_without_images: list,
    label_errors: list,
):
    recommendations = []

    if missing_labels:
        recommendations.append("พบรูปภาพที่ไม่มี label ควรตรวจสอบก่อน training")

    if labels_without_images:
        recommendations.append("พบ label ที่ไม่มีรูปภาพ对应กัน ควรลบหรือแก้ path")

    if label_errors:
        recommendations.append("พบ label format ผิด ควรแก้ก่อน training")

    if class_counter:
        max_count = max(class_counter.values())
        min_count = min(class_counter.values())

        if min_count > 0 and max_count / min_count >= 3:
            recommendations.append("Dataset มี class imbalance ควรเพิ่มข้อมูลใน class ที่มีจำนวนน้อย")

        for class_id, count in class_counter.items():
            if count < 50:
                recommendations.append(f"class {class_id} มี bbox น้อยกว่า 50 อาจทำให้ model เรียนรู้ไม่ดี")

    if not recommendations:
        recommendations.append("Dataset ดูพร้อมสำหรับเริ่ม training เบื้องต้น")

    return recommendations
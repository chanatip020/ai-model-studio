from pathlib import Path
import json


def load_json(path: Path) -> dict:
    if not path or not Path(path).exists():
        return {}

    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_recommendations(
    dataset_report_path: Path | None = None,
    training_result_path: Path | None = None,
):
    dataset_report = load_json(dataset_report_path) if dataset_report_path else {}
    training_result = load_json(training_result_path) if training_result_path else {}

    recommendations = []

    recommendations += analyze_dataset_report(dataset_report)
    recommendations += analyze_training_result(training_result)

    if not recommendations:
        recommendations.append({
            "level": "info",
            "category": "general",
            "title": "Dataset และ training result ดูพร้อมใช้งาน",
            "message": "ยังไม่พบปัญหาสำคัญจากข้อมูลที่มี",
            "action": "สามารถทดลอง train ต่อ หรือ export model ได้",
        })

    return {
        "summary": {
            "total_recommendations": len(recommendations),
            "critical": count_level(recommendations, "critical"),
            "warning": count_level(recommendations, "warning"),
            "info": count_level(recommendations, "info"),
        },
        "recommendations": recommendations,
    }


def count_level(recommendations: list, level: str) -> int:
    return sum(1 for item in recommendations if item.get("level") == level)


def analyze_dataset_report(report: dict) -> list:
    recs = []

    if not report:
        return recs

    summary = report.get("summary", {})
    issues = report.get("issues", {})
    class_distribution = report.get("class_distribution", {})

    health_score = summary.get("health_score", 100)
    total_images = summary.get("total_images", 0)
    total_bbox = summary.get("total_bbox", 0)

    if total_images < 100:
        recs.append({
            "level": "warning",
            "category": "dataset",
            "title": "จำนวนรูปภาพน้อยเกินไป",
            "message": f"พบรูปภาพทั้งหมด {total_images} รูป ซึ่งอาจยังไม่พอสำหรับ train model ให้แม่น",
            "action": "ควรเพิ่มข้อมูลอย่างน้อย 300–500 รูปต่อ use case ก่อน train จริง",
        })

    if total_bbox < 100:
        recs.append({
            "level": "warning",
            "category": "annotation",
            "title": "จำนวน bounding box น้อยเกินไป",
            "message": f"พบ bounding box ทั้งหมด {total_bbox} กล่อง",
            "action": "ควรเพิ่ม annotation โดยเฉพาะ defect/class ที่สำคัญ",
        })

    if health_score < 70:
        recs.append({
            "level": "critical",
            "category": "dataset",
            "title": "Dataset health score ต่ำ",
            "message": f"Health score ปัจจุบันคือ {health_score}",
            "action": "ควรแก้ label error, missing label และไฟล์ที่ไม่ตรงกันก่อน training",
        })

    missing_labels = issues.get("missing_labels", [])
    if missing_labels:
        recs.append({
            "level": "critical",
            "category": "label",
            "title": "พบรูปภาพที่ไม่มี label",
            "message": f"มีรูปภาพ {len(missing_labels)} ไฟล์ที่ไม่มี label",
            "action": "ตรวจสอบว่าเป็น OK image จริงหรือ label หาย ถ้าเป็น OK image ต้องกำหนด policy ให้ชัดเจน",
        })

    label_errors = issues.get("label_errors", [])
    if label_errors:
        recs.append({
            "level": "critical",
            "category": "label",
            "title": "พบ label format ผิด",
            "message": f"พบ label error จำนวน {len(label_errors)} จุด",
            "action": "แก้ไฟล์ label ให้เป็น YOLO format: class x_center y_center width height",
        })

    recs += analyze_class_balance(class_distribution)

    return recs


def analyze_class_balance(class_distribution: dict) -> list:
    recs = []

    if not class_distribution:
        return recs

    counts = []
    for class_id, info in class_distribution.items():
        counts.append({
            "class_id": class_id,
            "class_name": info.get("class_name", f"class_{class_id}"),
            "bbox_count": info.get("bbox_count", 0),
        })

    if not counts:
        return recs

    max_count = max(item["bbox_count"] for item in counts)
    min_count = min(item["bbox_count"] for item in counts)

    if min_count == 0:
        recs.append({
            "level": "critical",
            "category": "class_balance",
            "title": "บาง class ไม่มีข้อมูล",
            "message": "มี class ที่ไม่มี bounding box เลย",
            "action": "ต้องเพิ่ม sample หรือถอด class นั้นออกจาก training",
        })
        return recs

    imbalance_ratio = max_count / min_count

    if imbalance_ratio >= 3:
        weak_classes = [
            item for item in counts
            if item["bbox_count"] <= max_count / 3
        ]

        names = ", ".join(
            f"{item['class_name']}({item['bbox_count']})"
            for item in weak_classes
        )

        recs.append({
            "level": "warning",
            "category": "class_balance",
            "title": "พบ class imbalance",
            "message": f"สัดส่วน class มากสุด/น้อยสุด = {imbalance_ratio:.2f}. Class ที่น้อย: {names}",
            "action": "ควรเพิ่มรูปหรือ annotation ใน class ที่มีจำนวนน้อย เพื่อเพิ่ม recall ของ class นั้น",
        })

    for item in counts:
        if item["bbox_count"] < 50:
            recs.append({
                "level": "warning",
                "category": "class_sample",
                "title": f"Class {item['class_name']} มีข้อมูลน้อย",
                "message": f"Class นี้มี bbox เพียง {item['bbox_count']} กล่อง",
                "action": "ควรเพิ่มข้อมูล class นี้อย่างน้อย 100–300 bbox สำหรับ baseline model",
            })

    return recs


def analyze_training_result(result: dict) -> list:
    recs = []

    if not result:
        return recs

    metrics = result.get("metrics", {})

    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    map50 = metrics.get("mAP50", 0)
    map50_95 = metrics.get("mAP50_95", 0)

    if map50_95 and map50_95 < 0.5:
        recs.append({
            "level": "warning",
            "category": "model_quality",
            "title": "mAP50-95 ยังต่ำ",
            "message": f"mAP50-95 ปัจจุบันคือ {map50_95:.3f}",
            "action": "ควรเพิ่ม dataset, ตรวจ label quality, และลองเพิ่ม imgsz หรือ epochs",
        })

    if precision > 0 and recall > 0:
        if precision - recall >= 0.15:
            recs.append({
                "level": "warning",
                "category": "recall",
                "title": "Recall ต่ำกว่า Precision มาก",
                "message": f"Precision={precision:.3f}, Recall={recall:.3f}",
                "action": "โมเดลอาจตรวจไม่ครบ ควรเพิ่ม sample ที่ miss detection และลด confidence threshold ตอน inference",
            })

        if recall - precision >= 0.15:
            recs.append({
                "level": "warning",
                "category": "precision",
                "title": "Precision ต่ำกว่า Recall มาก",
                "message": f"Precision={precision:.3f}, Recall={recall:.3f}",
                "action": "โมเดลอาจ false alarm เยอะ ควรเพิ่ม negative sample และตรวจ label ที่คลุม object ไม่ถูกต้อง",
            })

    if map50 > 0 and map50_95 > 0:
        gap = map50 - map50_95

        if gap >= 0.25:
            recs.append({
                "level": "info",
                "category": "bbox_quality",
                "title": "mAP50 และ mAP50-95 ต่างกันมาก",
                "message": f"mAP50={map50:.3f}, mAP50-95={map50_95:.3f}",
                "action": "โมเดลอาจจับตำแหน่ง bbox ไม่แม่น ควรตรวจคุณภาพ label และลองเพิ่ม imgsz",
            })

    return recs
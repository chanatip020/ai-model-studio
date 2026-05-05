import json
from pathlib import Path
import pandas as pd

def parse_training_result(run_dir: Path):
    run_dir = Path(run_dir)

    results_csv = run_dir / "results.csv"
    summary_json = run_dir / "training_summary.json"

    if not results_csv.exists():
        return {"error": "results.csv not found"}

    df = pd.read_csv(results_csv)

    # เอาแถวสุดท้าย (final result)
    last_row = df.iloc[-1].to_dict()

    metrics = {
        "epoch": int(last_row.get("epoch", -1)),
        "train_loss": float(last_row.get("train/box_loss", 0)),
        "val_loss": float(last_row.get("val/box_loss", 0)),
        "precision": float(last_row.get("metrics/precision(B)", 0)),
        "recall": float(last_row.get("metrics/recall(B)", 0)),
        "mAP50": float(last_row.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(last_row.get("metrics/mAP50-95(B)", 0)),
    }

    # load summary
    summary = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

    result = {
        "run_name": summary.get("run_name"),
        "model": summary.get("model"),
        "dataset": summary.get("dataset_path"),
        "metrics": metrics,
        "paths": {
            "best_model": summary.get("best_model_path"),
            "result_dir": summary.get("result_path"),
        },
    }

    return result
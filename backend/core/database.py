import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()

TABLE_NAME = "experiments"


def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")

    return create_client(url, key)


def init_db():
    # Supabase ใช้ table ที่สร้างไว้แล้ว
    # function นี้คงไว้เพื่อให้ main.py ไม่ต้องแก้เยอะ
    return True


def save_experiment(parsed_result: dict):
    metrics = parsed_result.get("metrics", {})
    paths = parsed_result.get("paths", {})

    data = {
        "run_name": parsed_result.get("run_name"),
        "model": parsed_result.get("model"),
        "dataset_path": parsed_result.get("dataset"),
        "best_model_path": paths.get("best_model"),
        "result_path": paths.get("result_dir"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "map50": metrics.get("mAP50"),
        "map50_95": metrics.get("mAP50_95"),
        "metrics_json": metrics,
        "created_at": datetime.now().isoformat(),
    }

    response = get_client().table(TABLE_NAME).insert(data).execute()
    return response.data


def list_experiments(limit: int = 20):
    response = (
        get_client()
        .table(TABLE_NAME)
        .select("*")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data


def get_best_experiment():
    response = (
        get_client()
        .table(TABLE_NAME)
        .select("*")
        .order("map50_95", desc=True)
        .order("map50", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]
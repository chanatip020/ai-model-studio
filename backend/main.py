import argparse
from pathlib import Path

import json

from core.dataset_analyzer import analyze_yolo_dataset
from core.trainer import train_yolo_model
from core.result_parser import parse_training_result
from core.database import init_db, save_experiment, list_experiments, get_best_experiment
from core.recommendation_engine import generate_recommendations

def main():
    parser = argparse.ArgumentParser(description="AI Model Studio CLI")
    parser.add_argument("--mode", required=True, choices=["analyze", "train", "parse", "history", "best", "recommend"])
    parser.add_argument("--dataset", required=False, help="Path to YOLO dataset folder")
    parser.add_argument("--output", default="reports/dataset_report.json")

    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--device", default="0")

    parser.add_argument("--run_dir", help="Path to training run directory")

    parser.add_argument("--dataset_report", help="Path to dataset_report.json")
    parser.add_argument("--training_result", help="Path to parsed training result json")
    args = parser.parse_args()

    if args.mode == "analyze":
        report = analyze_yolo_dataset(
            dataset_dir=Path(args.dataset),
            output_path=Path(args.output),
        )

        print("\nDataset analysis completed")
        print(f"Images: {report['summary']['total_images']}")
        print(f"Labels: {report['summary']['total_labels']}")
        print(f"Classes: {report['summary']['total_classes']}")
        print(f"Report saved to: {args.output}")

    elif args.mode == "train":
        result = train_yolo_model(
            dataset_dir=Path(args.dataset),
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

        print("\nTraining completed")
        print(f"Run name: {result['run_name']}")
        print(f"Best model: {result['best_model_path']}")
        print(f"Result path: {result['result_path']}")

    elif args.mode == "parse":
        if not args.run_dir:
            raise ValueError("ต้องใส่ --run_dir")

        result = parse_training_result(Path(args.run_dir))
        save_experiment(result)

        print("\nParsed Result and saved to database:")
        print(json.dumps(result, indent=2, ensure_ascii=False))


    elif args.mode == "history":
        experiments = list_experiments(limit=20)

        print("\nExperiment History:")
        for exp in experiments:
            print(
                f"ID={exp['id']} | "
                f"Run={exp['run_name']} | "
                f"Model={exp['model']} | "
                f"mAP50={exp['map50']} | "
                f"mAP50-95={exp['map50_95']}"
            )

    elif args.mode == "best":
        best = get_best_experiment()

        print("\nBest Experiment:")
        print(json.dumps(best, indent=2, ensure_ascii=False))

    elif args.mode == "recommend":
        result = generate_recommendations(
            dataset_report_path=Path(args.dataset_report) if args.dataset_report else None,
            training_result_path=Path(args.training_result) if args.training_result else None,
        )

        output_path = Path("reports/recommendation_report.json")
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nRecommendation Report:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()
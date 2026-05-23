from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR
from src.data_utils import load_dataset, split_features_target
from src.modeling import build_models, build_preprocessor, evaluate_model, get_train_test_split


def _save_artifacts(
    *,
    best_name: str,
    best_pipeline,
    evaluation: dict,
    feature_columns: list[str],
    model_dir: Path,
) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.pkl"
    report_path = model_dir / "metrics.json"
    columns_path = model_dir / "feature_columns.json"

    joblib.dump(best_pipeline, model_path)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "metrics": evaluation}, f, indent=2)
    with columns_path.open("w", encoding="utf-8") as f:
        json.dump({"input_columns": feature_columns}, f, indent=2)

    return {
        "best_model": best_name,
        "model_path": str(model_path),
        "metrics_path": str(report_path),
        "feature_columns_path": str(columns_path),
        "metrics": evaluation,
    }


def train_from_dataframe(df: pd.DataFrame, model_dir: Path) -> dict:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    evaluation = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        evaluation[name] = metrics

        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained successfully.")

    return _save_artifacts(
        best_name=best_name,
        best_pipeline=best_pipeline,
        evaluation=evaluation,
        feature_columns=X.columns.tolist(),
        model_dir=model_dir,
    )


def train_and_save(data_path: Path, model_dir: Path) -> dict:
    df = load_dataset(str(data_path))
    return train_from_dataframe(df, model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train employee attrition models and save artifacts.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR, help="Directory to save model artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_and_save(args.data, args.output)

    metrics_df = pd.DataFrame(summary["metrics"]).T.sort_values(by="f1_score", ascending=False)
    print("Training complete.")
    print(f"Best model: {summary['best_model']}")
    print("\nModel comparison metrics:")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

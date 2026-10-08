import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

FEATURE_SCHEMA = "tfidf_word_char_ridge_v1"


def build_pipeline(alpha: float = 10.0) -> Pipeline:
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    transformer = ColumnTransformer(
        transformers=[
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True), "transcript"),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True), "transcript"),
        ],
        transformer_weights={"word": 1.0, "char": 0.5},
    )
    return Pipeline([("features", transformer), ("regressor", Ridge(alpha=alpha))])


def train_baseline(dataset_path: Path, output_dir: Path) -> dict:
    frame = pd.read_csv(dataset_path)
    required = {"participant_id", "split", "transcript", "phq8_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame[["participant_id", "split", "transcript", "phq8_score"]].isna().any().any():
        raise ValueError("Required participant, split, transcript, and PHQ fields must not be missing")
    if frame["participant_id"].duplicated().any():
        raise ValueError("Require one transcript row per participant; utterance/chunk rows are not independent PHQ labels")
    if not frame["split"].isin(["train", "dev", "test"]).all():
        raise ValueError("split must contain only train, dev, and test")
    if not frame["phq8_score"].between(0, 24).all():
        raise ValueError("PHQ-8 score target must be in the range 0–24")
    if not frame["transcript"].astype(str).str.strip().ne("").all():
        raise ValueError("Empty participant transcripts are not valid training samples")

    split_by_id = frame.groupby("participant_id")["split"].nunique()
    if (split_by_id != 1).any():
        raise ValueError("Participant leakage detected across splits")
    train = frame.loc[frame["split"] == "train"].copy()
    dev = frame.loc[frame["split"] == "dev"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if len(train) < 5 or len(dev) < 2 or len(test) < 1:
        raise ValueError("Need at least 5 train, 2 dev, and 1 test participant with PHQ labels")

    # Hyperparameters are fixed in advance. Dev is only a baseline gate; test is held out and never used here.
    model = build_pipeline()
    model.fit(train[["transcript"]], train["phq8_score"].astype(float))
    predictions = np.clip(model.predict(dev[["transcript"]]), 0.0, 24.0)
    median = float(train["phq8_score"].median())
    dev_mae = float(mean_absolute_error(dev["phq8_score"], predictions))
    baseline_mae = float(mean_absolute_error(dev["phq8_score"], np.full(len(dev), median)))
    dev_rmse = float(np.sqrt(mean_squared_error(dev["phq8_score"], predictions)))
    passes_dev_gate = bool(dev_mae < baseline_mae)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "phq8_text_baseline.joblib"
    manifest_path = output_dir / "phq8_text_baseline.manifest.json"
    joblib.dump(model, model_path)
    manifest = {
        "model_version": "phq8-text-ridge-v1",
        "task": "phq8_text_regression",
        "research_only": True,
        # An internal dev comparison does not authorize external or clinical deployment.
        "deployable": False,
        "passed_dev_baseline_gate": passes_dev_gate,
        "feature_schema": FEATURE_SCHEMA,
        "target": "official participant-level PHQ_Score",
        "n_train_participants": int(len(train)),
        "n_dev_participants": int(len(dev)),
        "n_test_participants": int(len(test)),
        "dev_mae": round(dev_mae, 4),
        "dev_rmse": round(dev_rmse, 4),
        "dev_median_baseline_mae": round(baseline_mae, 4),
        "test_split_used_for_selection": False,
        "warning": "Research baseline only. This manifest intentionally blocks hosted API serving; dev-baseline success is not clinical validation or deployment approval.",
    }
    manifest["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train a text-only participant-level E-DAIC PHQ-8 research baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/edaic_text.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()
    if args.alpha != 10.0:
        parser.error("alpha is fixed at 10.0 for this baseline; do not tune on the test set")
    manifest = train_baseline(args.dataset, args.output_dir)
    if not manifest["passed_dev_baseline_gate"]:
        raise SystemExit("Development MAE did not beat the train-median baseline. Serving remains disabled.")


if __name__ == "__main__":
    main()

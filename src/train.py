from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, classification_report

from src.bundle import build_training_matches, load_kaggle_bundle, ranking_snapshot_before
from src.features import build_training_frame
from src.model import FEATURE_COLUMNS, train_models


def time_based_train_test_split(df, test_size=0.2):
    frame = df.sort_values("date").reset_index(drop=True)
    split_index = max(1, int(len(frame) * (1 - test_size)))
    train_df = frame.iloc[:split_index].copy()
    test_df = frame.iloc[split_index:].copy()
    return train_df, test_df


def evaluate(df):
    frame = build_training_frame(df)
    train_df, test_df = time_based_train_test_split(frame, test_size=0.2)

    artifacts = train_models(train_df, output_dir=Path("artifacts"))

    from joblib import load

    classifier = load(artifacts.classifier_path)["model"]
    regressor = load(artifacts.regressor_path)["model"]

    X_test = test_df[FEATURE_COLUMNS]
    y_class = test_df["result"]
    y_reg = test_df["goal_diff"]

    class_probs = classifier.predict_proba(X_test)
    class_pred = classifier.predict(X_test)
    reg_pred = regressor.predict(X_test)

    return {
        "accuracy": accuracy_score(y_class, class_pred),
        "log_loss": log_loss(y_class, class_probs, labels=classifier.classes_),
        "mae": mean_absolute_error(y_reg, reg_pred),
        "report": classification_report(y_class, class_pred, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data_sources.toml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    bundle = load_kaggle_bundle(args.config)
    df = build_training_matches(bundle)
    artifacts = train_models(df, args.output)
    print(f"Saved classifier to {artifacts.classifier_path}")
    print(f"Saved regressor to {artifacts.regressor_path}")


if __name__ == "__main__":
    main()

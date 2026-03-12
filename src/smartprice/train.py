from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset, split_features_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a price prediction model.")
    parser.add_argument("--data", required=True, help="Path to CSV training data.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--model-out", required=True, help="Path to write model file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data)
    X, y = split_features_target(df, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": list(X.columns),
            "target": args.target,
        },
        model_path,
    )

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

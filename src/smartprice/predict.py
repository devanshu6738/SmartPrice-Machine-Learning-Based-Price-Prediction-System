from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate price predictions.")
    parser.add_argument("--data", required=True, help="Path to CSV data for prediction.")
    parser.add_argument("--model", required=True, help="Path to saved model file.")
    parser.add_argument("--out", required=True, help="Path to write predictions CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target = bundle["target"]

    df = load_dataset(args.data)
    features = df.reindex(columns=feature_columns)

    if features.isna().all(axis=None):
        raise ValueError("No matching feature columns found in prediction data.")

    preds = model.predict(features)

    out_df = df.copy()
    out_df[f"{target}_pred"] = preds

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()

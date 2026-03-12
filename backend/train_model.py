from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "price_model.pkl"

TARGET = "price"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical = ["brand"]
    numeric = [
        "ram",
        "storage",
        "processor_speed",
        "battery_capacity",
        "camera_mp",
    ]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ]
    )


def build_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "decision_tree": DecisionTreeRegressor(random_state=42),
    }


def train_and_evaluate() -> dict:
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    preprocessor = build_preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        results[name] = {
            "model": pipeline,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
        }

    return results


def pick_best(results: dict) -> tuple:
    best_name = max(results, key=lambda k: results[k]["r2"])
    return best_name, results[best_name]


def save_model(model, feature_columns) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
            },
            f,
        )


def main() -> None:
    results = train_and_evaluate()
    best_name, best = pick_best(results)

    print("Model comparison (higher R2 is better):")
    for name, metrics in results.items():
        print(
            f"- {name}: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}"
        )

    df = load_data()
    feature_columns = df.drop(columns=[TARGET]).columns.tolist()

    save_model(best["model"], feature_columns)
    print(f"Best model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

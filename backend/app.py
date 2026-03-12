from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "price_model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


class ModelNotReady(Exception):
    pass


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise ModelNotReady(
            "Model file not found. Train the model first: python backend/train_model.py"
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "brand",
        "ram",
        "storage",
        "processor_speed",
        "battery_capacity",
        "camera_mp",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    return {
        "brand": str(payload["brand"]),
        "ram": float(payload["ram"]),
        "storage": float(payload["storage"]),
        "processor_speed": float(payload["processor_speed"]),
        "battery_capacity": float(payload["battery_capacity"]),
        "camera_mp": float(payload["camera_mp"]),
    }


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        payload = request.get_json(force=True)
        features = validate_payload(payload)
        model_bundle = load_model()

        model = model_bundle["model"]
        feature_columns = model_bundle["feature_columns"]

        row = {col: features.get(col) for col in feature_columns}
        data = pd.DataFrame([row], columns=feature_columns)
        prediction = model.predict(data)[0]

        return jsonify({"predicted_price": round(float(prediction), 2)})
    except ModelNotReady as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/")
def index() -> Any:
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

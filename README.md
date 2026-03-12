# SmartPrice - Machine Learning Based Price Prediction System

SmartPrice is a full-stack ML web app that predicts the price of electronic products (smartphones or laptops) based on specs such as brand, RAM, storage, processor speed, battery capacity, and camera megapixels.

## Project Structure

SmartPrice/
+-- data/
¦   +-- dataset.csv
+-- model/
¦   +-- price_model.pkl
+-- notebooks/
¦   +-- training.ipynb
+-- backend/
¦   +-- app.py
¦   +-- train_model.py
+-- frontend/
¦   +-- index.html
¦   +-- style.css
¦   +-- script.js
+-- requirements.txt
+-- README.md

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Train the model

```bash
python backend/train_model.py
```

The script compares Linear Regression, Random Forest, and Decision Tree models, evaluates them using R2 score and MAE, and saves the best model to `model/price_model.pkl`.

## Run the Flask API

```bash
python backend/app.py
```

The API will run at `http://localhost:5000`.

### Example API request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand":"Samsung",
    "ram":8,
    "storage":128,
    "processor_speed":2.8,
    "battery_capacity":4500,
    "camera_mp":48
  }'
```

Response:

```json
{
  "predicted_price": 42000
}
```

## Frontend

Open `frontend/index.html` in a browser after starting the Flask server. The UI sends requests to `/predict` and displays the estimated price.

## Deploy on Render

1. Push this repo to GitHub.
2. In Render, create a new Web Service and connect the repo.
3. Render will detect `render.yaml` automatically. If not, set:
   - Build Command: `pip install -r requirements.txt && python backend/train_model.py`
   - Start Command: `gunicorn --chdir backend app:app`
4. Deploy. Use the generated URL as your base API URL.

## Optional Advanced Features
- Model comparison is included in `backend/train_model.py` output.
- Feature importance can be added by extending the training script to export plots.
- Deployment targets: Docker or Render.

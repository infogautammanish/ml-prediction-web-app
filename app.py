from flask import Flask, request, jsonify, render_template
import pickle
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
app = Flask(__name__)

with open(BASE / "model" / "house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        row = pd.DataFrame([{
            "area_sqft": float(data["area_sqft"]),
            "bedrooms": int(data["bedrooms"]),
            "bathrooms": int(data["bathrooms"]),
            "house_age": int(data["house_age"]),
            "location": data["location"]
        }])
        prediction = float(model.predict(row)[0])
        return jsonify({"success": True, "predicted_price": round(prediction, -3)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

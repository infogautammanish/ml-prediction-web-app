# 🔮 ML Prediction Web App

Full-stack house price prediction app using HTML/CSS/JavaScript, Flask, and Scikit-learn.

## Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python + Flask
- ML: Scikit-learn Random Forest Regressor
- Data: Pandas / NumPy

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```
Open `http://127.0.0.1:5000`.

## Flow
User Input → JavaScript → Flask API → Preprocessing → Random Forest → Prediction

## Note
The included housing dataset is synthetic and is provided to make the demo immediately runnable. Replace it with a properly sourced dataset for a production/portfolio version.

## Resume
**ML Prediction Web App | Python, Flask, JavaScript, Scikit-learn**
- Built an end-to-end house price prediction application using Random Forest regression.
- Developed a Flask prediction API and JavaScript frontend for real-time model inference.
- Implemented numerical and categorical preprocessing with Scikit-learn pipelines and OneHotEncoder.
- Evaluated the model using MAE and R² and integrated the trained model into a responsive web interface.

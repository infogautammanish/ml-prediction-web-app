import pandas as pd
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "housing.csv")
X, y = df.drop(columns=["price"]), df["price"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", ["area_sqft","bedrooms","bathrooms","house_age"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"])
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, max_depth=18, min_samples_leaf=2))
])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(f"MAE: ₹{mean_absolute_error(y_test,pred):,.0f}")
print(f"R²: {r2_score(y_test,pred):.3f}")
with open(BASE / "model" / "house_price_model.pkl","wb") as f:
    pickle.dump(model,f)
print("Model saved.")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/baseline.csv")

X = df[["amount", "txn_count", "age"]]
y = df["fraud"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/fraud_model.pkl")
print("Model trained and saved")

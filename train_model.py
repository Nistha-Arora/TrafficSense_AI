import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
import glob
import os

files = glob.glob("dataset/**/*.csv.gz", recursive=True)

if not files:
    raise FileNotFoundError(
        "Dataset folder ke andar CSV file nahi mili."
    )

print("Dataset found:", files[0])
df = pd.read_csv(files[0])
df = df.dropna().copy()

# Prepare date and time
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month

# Create traffic categories
low = df["traffic_volume"].quantile(0.33)
high = df["traffic_volume"].quantile(0.66)

df["traffic_level"] = "Medium"
df.loc[df["traffic_volume"] <= low, "traffic_level"] = "Low"
df.loc[df["traffic_volume"] > high, "traffic_level"] = "High"

features = [
    "hour", "day", "month",
    "temp", "rain_1h", "snow_1h", "clouds_all"
]

X = df[features]
y = df["traffic_level"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save model and feature information
joblib.dump(
    {"model": model, "features": features},
    "trafficsense_model.pkl"
)

print("Model saved successfully!")
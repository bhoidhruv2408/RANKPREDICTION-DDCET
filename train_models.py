"""
DDCET College Predictor AI - Model Training Script
Trains two models:
  1. Marks-to-Rank predictor (RandomForestRegressor)
  2. Saves college cutoff data for prediction logic
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "marks_vs_rank.csv"))
print(f"[INFO] Loaded {len(df)} rows from marks_vs_rank.csv")

# ── Encode category ────────────────────────────────────────────────────────
le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["category"])

# Save encoder so we can use it in app.py
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
print(f"[INFO] Categories found: {list(le.classes_)}")

# ── Features & target ──────────────────────────────────────────────────────
X = df[["year", "marks", "category_enc"]]
y = df["rank"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train RandomForestRegressor ────────────────────────────────────────────
print("[INFO] Training Rank Predictor model …")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"[RESULT] MAE  = {mae:.1f} ranks")
print(f"[RESULT] R²   = {r2:.4f}")

# ── Save model ─────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "rank_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"[INFO] Model saved → {model_path}")
print("[DONE] Training complete!")

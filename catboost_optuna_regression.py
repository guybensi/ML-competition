"""CatBoost + Optuna Regression Tuning (Colab Script)

This script:
1) Uploads train/test/sample_submission files in Colab.
2) Builds X/y and X_test (drops id/row identifiers).
3) Tunes a CatBoostRegressor with Optuna (objective = Median Absolute Error).
4) Trains a final CatBoost model on full training data.
5) Predicts on test, clips predictions to a safe range, validates outputs.
6) Writes a Kaggle-ready submission CSV.

Adjust FILES / TARGET_COL / ID_COLS / PRED_CLIP as needed.
"""
# =========================
# 0) Install dependencies
# =========================
!pip install -q catboost optuna scikit-learn pandas numpy


# =========================
# 1) Upload files in Colab
# =========================
from google.colab import files
uploaded = files.upload()

# =========================
# 2) Imports
# =========================
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error

from catboost import CatBoostRegressor


# =========================
# 3) Configuration
# =========================
FILES = {
    "train": "train (1).csv",
    "test": "test (1).csv",
    "sample_submission": "sample_submission (1).csv",  # optional, not strictly required
}

# Change these to match your dataset
TARGET_COL = "yield"          # e.g., "yield" or "Hardness" etc.
ID_COLS = ["id", "Row#"]      # identifier columns to drop (keep "id" in test for submission)

# If you want to restrict predictions to a known valid range, set this.
# Use None to disable clipping.
PRED_CLIP = (0.0, 10.0)

# Output file name
SUBMISSION_PATH = "submission_catboost_optuna_medae.csv"


# =========================
# 4) Load data
# =========================
train = pd.read_csv(FILES["train"])
test = pd.read_csv(FILES["test"])

# If you don't have Row# or id columns, errors='ignore' prevents crashes
X = train.drop(columns=ID_COLS + [TARGET_COL], errors="ignore")
y = train[TARGET_COL].copy()

X_test = test.drop(columns=ID_COLS, errors="ignore")

print(f"✅ Train X shape: {X.shape} | y shape: {y.shape}")
print(f"✅ Test X shape:  {X_test.shape}")


# =========================
# 5) Train/validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 6) Optuna objective
# =========================
def objective(trial: optuna.Trial) -> float:
    params = {
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_seed": 42,
        "loss_function": "MAE",
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    preds = model.predict(X_val)
    medae = median_absolute_error(y_val, preds)
    return medae


# =========================
# 7) Run tuning
# =========================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("✅ Best params:", study.best_params)
print("📉 Best MedAE:", study.best_value)


# =========================
# 8) Train final model
# =========================
final_model = CatBoostRegressor(
    iterations=1000,
    depth=study.best_params["depth"],
    learning_rate=study.best_params["learning_rate"],
    l2_leaf_reg=study.best_params["l2_leaf_reg"],
    random_strength=study.best_params["random_strength"],
    bagging_temperature=study.best_params["bagging_temperature"],
    loss_function="MAE",
    random_seed=42,
    verbose=100,
)

final_model.fit(X, y)


# =========================
# 9) Predict + validate
# =========================
test_preds = final_model.predict(X_test)

# Optional clipping
if PRED_CLIP is not None:
    lo, hi = PRED_CLIP
    test_preds = np.clip(test_preds, lo, hi)

# Basic sanity checks
assert len(test_preds) == len(test), "Prediction length does not match test rows."
assert not np.isnan(test_preds).any(), "Predictions contain NaN values."


# =========================
# 10) Build submission
# =========================
# Keep id from the original test file if present; otherwise create a simple index.
if "id" in test.columns:
    submission = pd.DataFrame({"id": test["id"].values})
else:
    submission = pd.DataFrame({"id": np.arange(len(test_preds))})

# IMPORTANT: set the correct submission column name for your competition.
# If your target is 'yield', use TARGET_COL; if Kaggle expects a different name, change here.
submission[TARGET_COL] = test_preds

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ Saved submission to: {SUBMISSION_PATH}")

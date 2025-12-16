
# ==================================
# Smoker Status Prediction - Training
# ==================================

# 0) Unzip dataset in Colab
import os
import zipfile

zip_path = "/content/apl-2025-spring-smoker-status.zip"
extract_path = "/content/"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted files:", os.listdir(extract_path))


# 1) Import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# 2) Load data
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
submission = pd.read_csv("/content/sample_submission.csv")


# 3) Exploratory Data Analysis
print("Target distribution (normalized):")
print(train["smoking"].value_counts(normalize=True))

sns.countplot(x="smoking", data=train)
plt.title("Class Distribution")
plt.tight_layout()
plt.show()


# 4) Preprocessing
X = train.drop(columns=["id", "smoking"])
y = train["smoking"]
X_test = test.drop(columns=["id"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)


# 5) Model benchmarking with Stratified K-Fold
models = {
    "CatBoost": CatBoostClassifier(verbose=0, random_seed=42, eval_metric="AUC"),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=1000),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_scores = {}

for name, model in models.items():
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X_processed, y):
        X_train_fold = X_processed[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X_processed[val_idx]

        model.fit(X_train_fold, y_train_fold)
        oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]

    auc = roc_auc_score(y, oof_preds)
    model_scores[name] = auc
    print(f"{name} ROC AUC: {auc:.5f}")


best_model = max(model_scores, key=model_scores.get)
print(f"Best model: {best_model} (AUC = {model_scores[best_model]:.5f})")

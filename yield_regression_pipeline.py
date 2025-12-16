"""Kaggle/Colab Regression Pipeline (Yield Prediction)

What this script does:
1) Loads train/test (+ sample submission) CSV files.
2) Drops identifier columns.
3) Removes near-zero variance features.
4) Optionally removes highly correlated features.
5) (Optional) Tries a small set of engineered features and keeps them if they improve CV MAE.
6) Benchmarks multiple models with 5-fold CV (MAE).
7) (Optional) Tunes RandomForest with Optuna.
8) Trains the best RandomForest configuration and writes a Kaggle-ready submission CSV.

Notes:
- This file is intended to run in Google Colab.
- If your CSV filenames are different, adjust FILES below.
"""
# =========================
# 0) Upload files in Colab
# =========================
from google.colab import files

# Upload: train.csv, test.csv, sample_submission.csv (or your variants)
uploaded = files.upload()

# =========================
# 1) Imports
# =========================
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

# Optional (if installed): XGBoost
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore")


# =========================
# 2) File configuration
# =========================
# Change these if your uploaded files have different names
FILES = {
    "train": "train (1).csv",
    "test": "test (1).csv",
    "sample_submission": "sample_submission (1).csv",
}

TARGET_COL = "yield"
ID_COLS = ["id", "Row#"]


# =========================
# 3) Utilities
# =========================
def load_data(files_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test and sample submission from CSV."""
    train_df = pd.read_csv(files_cfg["train"])
    test_df = pd.read_csv(files_cfg["test"])
    sub_df = pd.read_csv(files_cfg["sample_submission"])
    return train_df, test_df, sub_df


def drop_id_and_target(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_col: str, id_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split X/y and drop identifier columns."""
    X = train_df.drop(columns=id_cols + [target_col], errors="ignore")
    y = train_df[target_col].copy()
    X_test = test_df.drop(columns=id_cols, errors="ignore")
    return X, y, X_test


def remove_low_variance(X: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 1e-5):
    """Remove near-constant features using VarianceThreshold."""
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X)
    X_test_var = selector.transform(X_test)

    kept = X.columns[selector.get_support()].tolist()
    X_out = pd.DataFrame(X_var, columns=kept)
    X_test_out = pd.DataFrame(X_test_var, columns=kept)
    return X_out, X_test_out, kept


def remove_high_correlation(X: pd.DataFrame, X_test: pd.DataFrame, corr_threshold: float = 0.98):
    """Optionally drop highly correlated features (pairwise absolute correlation)."""
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]

    if to_drop:
        X = X.drop(columns=to_drop, errors="ignore")
        X_test = X_test.drop(columns=to_drop, errors="ignore")

    return X, X_test, to_drop


def cv_mae(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> tuple[float, float]:
    """Return mean and std MAE over KFold CV."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    mae = -scores.mean()
    std = scores.std()
    return mae, std


def try_engineered_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Create a few engineered features and keep them only if they improve CV MAE."""
    # Baseline
    base_model = RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=-1)
    base_mae, _ = cv_mae(base_model, X, y)
    print(f"✅ Base CV MAE (no engineered features): {base_mae:.4f}")

    X_eng = X.copy()

    # Pollinator-related features (if columns exist)
    bee_cols = ["honeybee", "bumbles", "andrena", "osmia"]
    if all(col in X_eng.columns for col in bee_cols):
        X_eng["bee_sum"] = X_eng[bee_cols].sum(axis=1)
        if "clonesize" in X_eng.columns:
            X_eng["bee_x_clone"] = X_eng["clonesize"] * X_eng["bee_sum"]
    else:
        print("ℹ️ Skipped bee_sum / bee_x_clone (missing columns).")

    # Temperature range (if columns exist)
    if "MaxOfUpperTRange" in X_eng.columns and "MinOfUpperTRange" in X_eng.columns:
        X_eng["temp_range"] = X_eng["MaxOfUpperTRange"] - X_eng["MinOfUpperTRange"]
    else:
        print("ℹ️ Skipped temp_range (missing columns).")

    eng_model = RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=-1)
    eng_mae, _ = cv_mae(eng_model, X_eng, y)
    print(f"🧪 CV MAE (with engineered features): {eng_mae:.4f}")

    if eng_mae < base_mae:
        print("✅ Keeping engineered features (improved CV MAE).")
        return X_eng
    else:
        print("⚠️ Discarding engineered features (no improvement).")
        return X


# =========================
# 4) Main pipeline
# =========================
def main():
    # Load
    train_df, test_df, sub_df = load_data(FILES)

    # Split
    X, y, X_test = drop_id_and_target(train_df, test_df, TARGET_COL, ID_COLS)
    print(f"✅ Training shape: {X.shape}, Test shape: {X_test.shape}")

    # Low variance filtering
    print("🔍 Removing near-zero variance features...")
    X, X_test, kept = remove_low_variance(X, X_test, threshold=1e-5)
    print(f"✅ Features after low-variance filtering: {X.shape[1]}")

    # High correlation filtering (optional)
    X, X_test, dropped_corr = remove_high_correlation(X, X_test, corr_threshold=0.98)
    if dropped_corr:
        print(f"⚠️ Dropped {len(dropped_corr)} highly correlated features.")
    print(f"✅ Final number of features: {X.shape[1]}")

    # Feature engineering (optional keep-if-better)
    print("✨ Trying a small set of engineered features...")
    X_final = try_engineered_features(X, y)

    # Align test columns if engineered features were kept
    X_test_final = X_test.copy()
    for col in X_final.columns:
        if col not in X_test_final.columns:
            # Create missing engineered features for test, if possible
            if col == "bee_sum":
                bee_cols = ["honeybee", "bumbles", "andrena", "osmia"]
                if all(c in X_test_final.columns for c in bee_cols):
                    X_test_final[col] = X_test_final[bee_cols].sum(axis=1)
                else:
                    X_test_final[col] = 0.0
            elif col == "bee_x_clone":
                if "clonesize" in X_test_final.columns and "bee_sum" in X_test_final.columns:
                    X_test_final[col] = X_test_final["clonesize"] * X_test_final["bee_sum"]
                else:
                    X_test_final[col] = 0.0
            elif col == "temp_range":
                if "MaxOfUpperTRange" in X_test_final.columns and "MinOfUpperTRange" in X_test_final.columns:
                    X_test_final[col] = X_test_final["MaxOfUpperTRange"] - X_test_final["MinOfUpperTRange"]
                else:
                    X_test_final[col] = 0.0
            else:
                X_test_final[col] = 0.0

    # Ensure same column order
    X_test_final = X_test_final[X_final.columns]

    # Benchmark models
    print("
🔍 Benchmarking models with 5-fold CV (MAE)...")
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(random_state=42)

    results = {}
    for name, model in models.items():
        mae, std = cv_mae(model, X_final, y)
        results[name] = mae
        print(f"✅ {name}: MAE = {mae:.4f} ± {std:.4f}")

    top3 = sorted(results.items(), key=lambda x: x[1])[:3]
    print("
🏆 Top models:")
    for name, mae in top3:
        print(f"- {name}: MAE = {mae:.4f}")

    # Optional: Optuna tuning for RandomForest (uncomment to run)
    # ----------------------------------------------------------
    # !pip -q install optuna
    # import optuna
    #
    # def objective(trial):
    #     params = {
    #         "n_estimators": trial.suggest_int("n_estimators", 100, 600),
    #         "max_depth": trial.suggest_int("max_depth", 3, 20),
    #         "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
    #         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
    #         "random_state": 42,
    #         "n_jobs": -1,
    #     }
    #     model = RandomForestRegressor(**params)
    #     mae, _ = cv_mae(model, X_final, y)
    #     return mae
    #
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=30)
    # print("✅ Best RF MAE:", study.best_value)
    # print("📌 Best RF params:", study.best_params)

    # Train final model (using your tuned RF parameters)
    best_rf = RandomForestRegressor(
        n_estimators=347,
        max_depth=9,
        min_samples_split=4,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    best_rf.fit(X_final, y)

    # Predict and write submission
    preds = best_rf.predict(X_test_final)

    submission = sub_df.copy()
    submission[TARGET_COL] = preds
    out_path = "submission_best_rf.csv"
    submission.to_csv(out_path, index=False)
    print(f"
✅ Submission saved to: {out_path}")


if __name__ == "__main__":
    main()

import os
import sys
import json
import joblib
import pandas as pd
import lightgbm as lgb

from fraud_detection.configuration.logger import logging
from fraud_detection.configuration.exception import MyException

# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = "proj_data/processed"
DATA_PATH = os.path.join(PROCESSED_DIR, "final_df_clean.parquet")

MODEL_ROOT = "proj_data/models"
FULL_DIR = os.path.join(MODEL_ROOT, "full")
TOP_DIR = os.path.join(MODEL_ROOT, "top100")

for d in [MODEL_ROOT, FULL_DIR, TOP_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# LIGHTGBM PARAMS (FROM YOUR EXPERIMENTS)
# ============================================================

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 27.46147358274595,
    "n_jobs": 1,
    "random_state": 42
}

TOP_N_FEATURES = 100

# ============================================================
# MODEL BUILDING
# ============================================================

def build_models():
    try:
        logging.info("========== MODEL BUILDING STARTED ==========")

        # --------------------------------------------------
        # Step 1: Load processed dataset
        # --------------------------------------------------
        df = pd.read_parquet(DATA_PATH)
        logging.info(f"Dataset shape: {df.shape}")

        # --------------------------------------------------
        # Step 2: Convert object/string → category
        # --------------------------------------------------
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ["isFraud", "TransactionID"]]

        for col in cat_cols:
            df[col] = df[col].astype("category")

        logging.info(f"Categorical columns converted: {len(cat_cols)}")

        # --------------------------------------------------
        # Step 3: Time-based train-test split (NO LEAKAGE)
        # --------------------------------------------------
        df = df.sort_values("TransactionDT")
        split_point = df["TransactionDT"].quantile(0.8)

        train_df = df[df["TransactionDT"] <= split_point]
        test_df  = df[df["TransactionDT"] > split_point]

        X_train = train_df.drop(columns=["isFraud", "TransactionID"])
        y_train = train_df["isFraud"]

        X_test = test_df.drop(columns=["isFraud", "TransactionID"])
        y_test = test_df["isFraud"]

        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")

        # --------------------------------------------------
        # Step 4: Encode categorical columns (Arrow-safe)
        # --------------------------------------------------
        cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
        logging.info(f"Encoding {len(cat_cols)} categorical columns")

        for col in cat_cols:
            X_train[col] = X_train[col].cat.codes
            X_test[col] = X_test[col].cat.codes

        Xtr = X_train.astype("float32")
        Xte = X_test.astype("float32")

        # --------------------------------------------------
        # Step 5: Train FULL model (BASELINE – ONLY ONCE)
        # --------------------------------------------------
        full_model_path = os.path.join(FULL_DIR, "model.pkl")

        if not os.path.exists(full_model_path):
            logging.info("Training FULL feature LightGBM (baseline)")

            model_full = lgb.LGBMClassifier(**LGB_PARAMS)
            model_full.fit(Xtr, y_train)

            joblib.dump(model_full, full_model_path)

            Xtr.to_parquet(os.path.join(FULL_DIR, "X_train.parquet"))
            Xte.to_parquet(os.path.join(FULL_DIR, "X_test.parquet"))
            y_train.to_frame("isFraud").to_parquet(os.path.join(FULL_DIR, "y_train.parquet"))
            y_test.to_frame("isFraud").to_parquet(os.path.join(FULL_DIR, "y_test.parquet"))

            # Feature importance (baseline)
            fi = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model_full.feature_importances_
            }).sort_values("importance", ascending=False)

            fi["importance_pct"] = fi["importance"] / fi["importance"].sum()
            fi["cumulative_importance"] = fi["importance_pct"].cumsum()

            fi.to_csv(os.path.join(MODEL_ROOT, "feature_importance.csv"), index=False)

            top_features = fi["feature"].iloc[:TOP_N_FEATURES].tolist()

            with open(os.path.join(MODEL_ROOT, "top_100_features.json"), "w") as f:
                json.dump(top_features, f, indent=4)

            logging.info("FULL baseline model trained and frozen")

        else:
            logging.info("FULL model already exists. Skipping baseline training.")

            with open(os.path.join(MODEL_ROOT, "top_100_features.json")) as f:
                top_features = json.load(f)

        # --------------------------------------------------
        # Step 6: Train TOP-100 model (PRODUCTION MODEL)
        # --------------------------------------------------
        logging.info("Training TOP-100 feature LightGBM")

        Xtr_top = Xtr[top_features]
        Xte_top = Xte[top_features]

        model_top = lgb.LGBMClassifier(**LGB_PARAMS)
        model_top.fit(Xtr_top, y_train)

        joblib.dump(model_top, os.path.join(TOP_DIR, "model.pkl"))
        Xtr_top.to_parquet(os.path.join(TOP_DIR, "X_train.parquet"))
        Xte_top.to_parquet(os.path.join(TOP_DIR, "X_test.parquet"))
        y_train.to_frame("isFraud").to_parquet(os.path.join(TOP_DIR, "y_train.parquet"))
        y_test.to_frame("isFraud").to_parquet(os.path.join(TOP_DIR, "y_test.parquet"))

        logging.info("========== MODEL BUILDING COMPLETED ==========")

    except Exception as e:
        logging.error("MODEL BUILDING FAILED", exc_info=True)
        raise MyException(e, sys)


if __name__ == "__main__":
    build_models()
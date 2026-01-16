import os
import sys
import json
import pandas as pd
import pyarrow.parquet as pq

from fraud_detection.configuration.logger import logging
from fraud_detection.configuration.exception import MyException


# ============================================================
# PATHS
# ============================================================

RAW_DIR = "proj_data/raw"
PROCESSED_DIR = "proj_data/processed"
ARTIFACT_DIR = "proj_data/artifacts"

VALIDATION_DIR = os.path.join(ARTIFACT_DIR, "data_validation")
SCHEMA_DIR = os.path.join(ARTIFACT_DIR, "schema")
PREPROCESS_DIR = os.path.join(ARTIFACT_DIR, "preprocessing")

for d in [PROCESSED_DIR, VALIDATION_DIR, SCHEMA_DIR, PREPROCESS_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# UTILS
# ============================================================

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
    logging.info(f"Artifact saved at {path}")


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def preprocess():
    try:
        logging.info("========== DATA PREPROCESSING STARTED ==========")

        # --------------------------------------------------
        # STEP 1: Load ingested data
        # --------------------------------------------------
        logging.info("Step 1: Loading ingested parquet files")

        train_tr = pd.read_parquet(os.path.join(RAW_DIR, "train_transaction.parquet"))
        train_id = pd.read_parquet(os.path.join(RAW_DIR, "train_identity.parquet"))

        logging.info(f"train_transaction shape: {train_tr.shape}")
        logging.info(f"train_identity shape: {train_id.shape}")

        # --------------------------------------------------
        # STEP 2: Merge transaction + identity
        # --------------------------------------------------
        logging.info("Step 2: Merging on TransactionID")

        df = train_tr.merge(train_id, on="TransactionID", how="left")
        logging.info(f"Merged dataframe shape: {df.shape}")

        save_json(
            {"rows": df.shape[0], "columns": df.shape[1]},
            os.path.join(VALIDATION_DIR, "merged_shape.json")
        )

        # --------------------------------------------------
        # STEP 3: Missing value & dtype report
        # --------------------------------------------------
        logging.info("Step 3: Generating missing value & dtype report")

        mv_report = pd.DataFrame({
            "missing_pct": round(df.isnull().mean() * 100, 2),
            "dtype": df.dtypes.astype(str)
        })

        mv_report.to_csv(
            os.path.join(VALIDATION_DIR, "missing_value_report.csv")
        )

        df.head(2).T.to_csv(
            os.path.join(VALIDATION_DIR, "sample_rows.csv")
        )

        # --------------------------------------------------
        # STEP 4: Single-unique columns
        # --------------------------------------------------
        logging.info("Step 4: Checking single-unique columns")

        single_unique_cols = [c for c in df.columns if df[c].nunique() == 1]
        logging.info(f"Single unique columns count: {len(single_unique_cols)}")

        save_json(
            single_unique_cols,
            os.path.join(VALIDATION_DIR, "single_unique_columns.json")
        )

        # --------------------------------------------------
        # STEP 5: High missing columns (>30%)
        # --------------------------------------------------
        logging.info("Step 5: Identifying high-missing columns (>30%)")

        high_missing_cols = [
            c for c in df.columns if df[c].isnull().mean() > 0.30
        ]

        logging.info(f"High missing columns count: {len(high_missing_cols)}")

        save_json(
            high_missing_cols,
            os.path.join(VALIDATION_DIR, "high_missing_columns.json")
        )

        # --------------------------------------------------
        # STEP 6: Identify numeric vs categorical columns
        # --------------------------------------------------
        logging.info("Step 6: Detecting numeric and categorical columns")

        numeric_cols, categorical_cols = [], []

        for col in df.columns:
            if col in ["TransactionID", "isFraud"]:
                continue

            coerced = pd.to_numeric(df[col], errors="coerce")

            # If ANY numeric signal exists → numeric
            if coerced.notna().sum() > 0:
                df[col] = coerced
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        logging.info(f"Numeric columns count: {len(numeric_cols)}")
        logging.info(f"Categorical columns count: {len(categorical_cols)}")

        save_json(numeric_cols, os.path.join(SCHEMA_DIR, "numeric_columns.json"))
        save_json(categorical_cols, os.path.join(SCHEMA_DIR, "categorical_columns.json"))

        # --------------------------------------------------
        # STEP 7: Imputation
        # --------------------------------------------------
        logging.info("Step 7: Applying imputation")

        numeric_cols = [c for c in numeric_cols if c != "TransactionID"]
        categorical_cols = [c for c in categorical_cols if c != "TransactionID"]

        df[numeric_cols] = df[numeric_cols].fillna(-1)
        df[categorical_cols] = df[categorical_cols].fillna("unknown")

        save_json(
            {"numeric": -1, "categorical": "unknown"},
            os.path.join(PREPROCESS_DIR, "imputation_rules.json")
        )

        # --------------------------------------------------
        # STEP 8: Downcasting numeric columns
        # --------------------------------------------------
        logging.info("Step 8: Downcasting numeric columns")

        downcast_map = {}

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")
            downcast_map[col] = str(df[col].dtype)

        save_json(
            downcast_map,
            os.path.join(PREPROCESS_DIR, "downcast_dtypes.json")
        )

        # --------------------------------------------------
        # STEP 9: Convert categoricals to category dtype
        # --------------------------------------------------
        logging.info("Step 9: Converting categorical columns to category dtype")

        for col in categorical_cols:
            df[col] = df[col].astype("category")

        logging.info(f"Categorical columns converted: {len(categorical_cols)}")

        # --------------------------------------------------
        # STEP 10: Final safety check
        # --------------------------------------------------
        logging.info("Step 10: Validating numeric columns")

        bad_numeric = [
            col for col in numeric_cols
            if df[col].dtype == "object"
        ]

        if bad_numeric:
            raise ValueError(
                f"Numeric columns contain strings: {bad_numeric[:5]}"
            )

        # --------------------------------------------------
        # STEP 11: Save processed dataset
        # --------------------------------------------------
        logging.info("Step 11: Saving processed dataset")

        final_path = os.path.join(PROCESSED_DIR, "final_df.parquet")
        df.to_parquet(
            final_path,
            engine="pyarrow",
            compression="zstd",
            index=False
        )

        table = pq.read_table(final_path)
        pq.write_table(
            table,
            os.path.join(PROCESSED_DIR, "final_df_clean.parquet"),
            compression="zstd"
        )

        logging.info("========== DATA PREPROCESSING COMPLETED ==========")

    except Exception as e:
        logging.error("DATA PREPROCESSING FAILED", exc_info=True)
        raise MyException(e, sys)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess()
import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.configuration.logger import logging
from fraud_detection.configuration.exception import MyException


# ============================================================
# INGESTION UTILITIES (NO LOGGING OF ERRORS HERE)
# ============================================================

def csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    """
    Read CSV and write Parquet.
    This function ONLY ingests data, no transformations.
    """
    try:
        logging.info(f"Reading CSV: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logging.info(f"Loaded data shape: {df.shape}")

        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, parquet_path, compression="zstd")

        logging.info(f"Saved Parquet to: {parquet_path}")

    except Exception as e:
        # IMPORTANT: no logging here, just raise
        raise MyException(e, sys) from e


# ============================================================
# MAIN EXECUTION (PIPELINE BOUNDARY)
# ============================================================

def main():
    """
    Ingestion stage:
    raw_data → proj_data/raw (Parquet)
    """
    RAW_DIR = "raw_data_location"  # <-- adjust if needed
    OUT_DIR = "proj_data/raw"

    files = {
        "train_transaction": "train_transaction.csv",
        "train_identity": "train_identity.csv",
        "test_transaction": "test_transaction.csv",
        "test_identity": "test_identity.csv",
    }

    logging.info("Starting data ingestion stage")

    try:
        for name, file in files.items():
            csv_path = os.path.join(RAW_DIR, file)
            parquet_path = os.path.join(OUT_DIR, f"{name}.parquet")

            logging.info(f"Ingesting {name}")
            csv_to_parquet(csv_path, parquet_path)

        logging.info("Data ingestion stage completed successfully")

    except Exception:
        # Log ONCE, with full traceback
        logging.exception("❌ Data ingestion pipeline failed")
        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
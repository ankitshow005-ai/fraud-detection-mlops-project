import os
import sys
import json
import shutil
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn
import dagshub

from fraud_detection.configuration.logger import logging
from fraud_detection.configuration.exception import MyException


# ============================================================
# CONFIG
# ============================================================

MODEL_ROOT = "proj_data/models"

FULL_DIR = os.path.join(MODEL_ROOT, "full")
TOP_DIR = os.path.join(MODEL_ROOT, "top100")
APPROVED_DIR = os.path.join(MODEL_ROOT, "approved")

os.makedirs(APPROVED_DIR, exist_ok=True)

EXPERIMENT_NAME = "fraud-model-evaluation-new"

THRESHOLD = 0.4
APPROVAL_METRIC = "recall"
MIN_IMPROVEMENT = 0.01   # 1% improvement required


# ============================================================
# MLFLOW (KEEP SAME AS YOUR PREVIOUS SETUP)
# ============================================================

mlflow.set_tracking_uri(
    "https://dagshub.com/ankitshow005/fraud-detection-mlops-project.mlflow"
)

dagshub.init(
    repo_owner="ankitshow005",
    repo_name="fraud-detection-mlops-project",
    mlflow=True
)


# ============================================================
# UTILS
# ============================================================

def evaluate(model, X, y, threshold):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs)
    }


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_previous_top_metrics():
    metrics_path = os.path.join(APPROVED_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ============================================================
# MODEL EVALUATION
# ============================================================

def run_evaluation():
    try:
        logging.info("========== MODEL EVALUATION STARTED ==========")
        mlflow.set_experiment(EXPERIMENT_NAME)

        # --------------------------------------------------
        # FULL MODEL (BASELINE)
        # --------------------------------------------------
        with mlflow.start_run(run_name="full_model"):

            model_full = joblib.load(os.path.join(FULL_DIR, "model.pkl"))
            X_full = pd.read_parquet(os.path.join(FULL_DIR, "X_test.parquet"))
            y_full = pd.read_parquet(
                os.path.join(FULL_DIR, "y_test.parquet")
            ).values.ravel()

            full_metrics = evaluate(model_full, X_full, y_full, THRESHOLD)
            save_json(full_metrics, os.path.join(FULL_DIR, "metrics.json"))

            mlflow.log_param("model_type", "full")
            mlflow.log_param("threshold", THRESHOLD)

            for k, v in full_metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(
                model_full,
                name="full_model"
            )

        # --------------------------------------------------
        # TOP-100 MODEL (VERSIONED)
        # --------------------------------------------------
        with mlflow.start_run(run_name="top100_model"):

            model_top = joblib.load(os.path.join(TOP_DIR, "model.pkl"))
            X_top = pd.read_parquet(os.path.join(TOP_DIR, "X_test.parquet"))
            y_top = pd.read_parquet(
                os.path.join(TOP_DIR, "y_test.parquet")
            ).values.ravel()

            top_metrics = evaluate(model_top, X_top, y_top, THRESHOLD)
            save_json(top_metrics, os.path.join(TOP_DIR, "metrics.json"))

            mlflow.log_param("model_type", "top100")
            mlflow.log_param("threshold", THRESHOLD)

            for k, v in top_metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(
                model_top,
                name="top100_model"
            )

        # --------------------------------------------------
        # APPROVAL LOGIC (TOP100 vs PREVIOUS TOP100)
        # --------------------------------------------------
        prev_metrics = load_previous_top_metrics()
        approved = False

        if prev_metrics is None:
            approved = True
        else:
            approved = (
                top_metrics[APPROVAL_METRIC]
                >= prev_metrics[APPROVAL_METRIC] + MIN_IMPROVEMENT
            )

        approval_report = {
            "approval_metric": APPROVAL_METRIC,
            "threshold": THRESHOLD,
            "current_top100_score": top_metrics[APPROVAL_METRIC],
            "previous_top100_score": prev_metrics[APPROVAL_METRIC]
            if prev_metrics else None,
            "approved": approved
        }

        save_json(
            approval_report,
            os.path.join(APPROVED_DIR, "approval.json")
        )

        if approved:
            logging.info("TOP100 model APPROVED for production")

            shutil.copy(
                os.path.join(TOP_DIR, "model.pkl"),
                os.path.join(APPROVED_DIR, "model.pkl")
            )

            save_json(
                top_metrics,
                os.path.join(APPROVED_DIR, "metrics.json")
            )

        logging.info("========== MODEL EVALUATION COMPLETED ==========")

    except Exception as e:
        logging.error("MODEL EVALUATION FAILED", exc_info=True)
        raise MyException(e, sys)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_evaluation()
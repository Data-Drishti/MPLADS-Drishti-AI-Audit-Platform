import os
from pathlib import Path
import sys
import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
  sys.path.insert(0, str(BACKEND_DIR))

from engines.financialAndAnomalyCore import build_auditor_pipeline

DATASET_PATH = BACKEND_DIR / "Datasets" / "mplads_expenditures_2026-09-17.csv"
MODEL_DIR = BACKEND_DIR / "models"
ARTIFACT_PATH = MODEL_DIR / "iforest_pipeline.joblib"


def train_offline():
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  df = pd.read_csv(DATASET_PATH)

  X_train = df[["Expenditure Amount (₹)"]]
  pipeline = build_auditor_pipeline(contamination=0.03)
  pipeline.fit(X_train)

  joblib.dump(pipeline, ARTIFACT_PATH)

if __name__ == "__main__":
  train_offline()
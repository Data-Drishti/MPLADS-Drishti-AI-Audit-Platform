from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "iforest_pipeline.joblib"

class AmountDataCleaner(BaseEstimator, TransformerMixin):
  """
  Cleans raw monetary data:
  Strips commas, whitespace, leading zero padding, handles NaNs,
  and fractional amount inputs.
  Always returns a 2D NumPy array of shape (N, 1).
  """

  def __init__(self, min_threshold: float = 1.0, default_fill: float = 0.0):
    self.min_threshold = min_threshold
    self.default_fill = default_fill

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    amounts = np.asarray(X).ravel()
    cleaned = []

    for val in amounts:
      try:
        if isinstance(val, str):
          val = val.replace(",", "").strip()
        num = float(val)
      except (ValueError, TypeError):
        cleaned.append(self.default_fill)
        continue

      if np.isnan(num) or num < self.min_threshold:
        cleaned.append(self.default_fill)
        continue

      cleaned.append(num)

    return np.array(cleaned, dtype=np.float64).reshape(-1, 1)


class BenfordSurprisalTransformer(BaseEstimator, TransformerMixin):
  """
  Extracts first non-zero digit d in {1..9} and computes 
  normalized surprisal:
  S(d) = (I(d) - I_min) / (I_max - I_min).
  Outputs a 2D array of shape (N, 2): [leading_digit, normalized_surprisal].
  """

  def __init__(self):
    self.I_min = -np.log2(np.log10(1.0 + (1.0 / 1.0)))
    self.I_max = -np.log2(np.log10(1.0 + (1.0 / 9.0)))

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    amounts = np.asarray(X).ravel()
    surprisals = []

    for val in amounts:
      if val < 1.0:
        surprisals.append([0.0, 0.0])
        continue

      d1 = int(str(int(val))[0])
      p_d = np.log10(1.0 + (1.0 / d1))
      i_d = -np.log2(p_d)
      norm_i_d = (i_d - self.I_min) / (self.I_max - self.I_min)
      surprisals.append([float(d1), float(norm_i_d)])

    return np.array(surprisals, dtype=np.float64)


def build_auditor_pipeline(contamination: float = 0.03) -> Pipeline:
  """
  Constructs the Scikit-Learn Pipeline:
  Cleaner -> Feature Union (Log1p + Benford) -> Isolation Forest.
  """
  feature_union = ColumnTransformer(
      transformers=[
          ("log_amount", FunctionTransformer(np.log1p, validate=False), [0]),
          ("benford", BenfordSurprisalTransformer(), [0]),
      ],
      remainder="drop",
  )

  return Pipeline([
      ("cleaner", AmountDataCleaner(min_threshold=1.0, default_fill=0.0)),
      ("features", feature_union),
      ("isolation_forest",IsolationForest(
                                        n_estimators=150,
                                        contamination=contamination,
                                        random_state=42,
                                        n_jobs=-1)
        )
    ])

class AuditingEngine:
  """
  Final auditing engine:
  Analyses expenditures to flag potentially tmapered transactions.
  """

  def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
    self.pipeline=joblib.load(model_path)
    self.cleaner=self.pipeline.named_steps["cleaner"]
    self.features=self.pipeline.named_steps["features"]
    self.model=self.pipeline.named_steps["isolation_forest"]


  def audit_amount(self, amount, threshold: int = 1000000):
    df=pd.DataFrame([{'Expenditure Amount (₹)': amount}])
    clean_data=self.cleaner.transform(df[['Expenditure Amount (₹)']])
    amt=float(clean_data[0,0])

    if amt<1.0:
      return {
        "engine": "Financial & Anomaly Core",
        "sub_score": 0.0,
        "risk_tier": "NA",
        "metrics": {"amount": amount, "status": "INVALID_OR_NOISE"},
        "anomalies": ["Amount is non-positive, NaN, or sub-rupee noise."]
      }

    features_matrix=self.features.transform(clean_data)
    d1=int(features_matrix[0,1])
    benford_surprisal=float(features_matrix[0,2])

    score=float(self.model.decision_function(features_matrix)[0])
    if score >= 0.0:
      norm_inlier = max(0.0, 1.0 - (score / 0.218))
      s_iforest = float(0.35 * norm_inlier)
    else:
      severity = min(1.0, abs(score) / 0.120)
      s_iforest = float(0.35 + (0.65 * severity))

    composite_score=(0.35 * benford_surprisal) + (0.65 * s_iforest)
    sub_score = round(min(100.0, composite_score * 100.0), 1)

    if sub_score < 30.0:
      tier = "GREEN"
    elif sub_score <= 70.0:
      tier = "AMBER"
    else:
      tier = "RED"

    anomalies = []
    if s_iforest > 0.65:
      anomalies.append(
          f"Statistical Cost Outlier: Flagged by Isolation Forest (Decision"
          f" Score: {score:.3f})."
      )
    if benford_surprisal > 0.85:
      anomalies.append(
          f"Digit Irregularity: Rare leading digit '{d1}' exhibits high Benford"
          f" surprisal ({benford_surprisal:.2f})."
      )

    return {
        "engine": "Financial & Anomaly Core",
        "sub_score": sub_score,
        "risk_tier": tier,
        "metrics": {
            "sanitized_amount": amt,
            "leading_digit": d1,
            "benford_surprisal": round(benford_surprisal, 3),
            "iforest_score": round(s_iforest, 3),
        },
        "anomalies": (
            anomalies
            if anomalies
            else ["All financial parameters within standard limits."]
        ),
    }
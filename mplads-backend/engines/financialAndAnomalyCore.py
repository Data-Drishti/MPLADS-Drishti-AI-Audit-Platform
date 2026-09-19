from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class AmountDataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans Amount data feature to make them computable in analysis.
    """
    def __init__(self, min_threshold: float = 1.0, default_fill: float = 0.0):
        self.minimum_val = min_threshold
        self.default_val = default_fill
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
                cleaned.append(self.default_val)
                continue

            if np.isnan(num) or num < self.minimum_val:
                cleaned.append(self.default_val)
                continue

            cleaned.append(num)

        return np.array(cleaned, dtype=np.float64).reshape(-1, 1)

"""
Dataset Preprocessing and Preparation Module (spec Section 6.5).

Prepares Term 1 feature vectors and targets for training:
- Regressor target: final_exam_overall (0-100)
- Classifier target: risk_category (0: On Track, 1: Needs Monitoring, 2: Needs Support)
- Train/test split (80/20) stratified for classifier
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.ml.features import FEATURE_NAMES, extract_features_from_dict
from app.ml.data_generation import generate_dataset, SEED, TERM1

LABEL_MAP = {
    "On Track": 0,
    "Needs Monitoring": 1,
    "Needs Support": 2,
}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def prepare_training_dataset(seed: int = SEED) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Builds the Term 1 training dataset from the synthetic generator.
    Returns:
        X (DataFrame, 350x24)
        y_reg (Series of float final scores)
        y_clf (Series of int class indices 0, 1, 2)
    """
    raw_ds = generate_dataset(seed=seed)
    feature_rows = []
    y_reg_list = []
    y_clf_list = []

    for st in raw_ds["students"]:
        t1 = st["terms"][0]  # Term 1
        feats = extract_features_from_dict(t1)
        feature_rows.append(feats)

        final_score = t1.get("final_overall", 60.0)
        risk_str = t1.get("risk_label", "On Track")

        y_reg_list.append(float(final_score))
        y_clf_list.append(LABEL_MAP[risk_str])

    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    y_reg = pd.Series(y_reg_list, name="final_score")
    y_clf = pd.Series(y_clf_list, name="risk_category")

    return X, y_reg, y_clf


def get_train_test_splits(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
    test_size: float = 0.20,
    random_state: int = SEED,
) -> Dict[str, Any]:
    """Stratified 80/20 train/test split based on classifier categories."""
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_clf,
        random_state=random_state,
    )

    return {
        "X_train": X.iloc[train_idx].reset_index(drop=True),
        "X_test": X.iloc[test_idx].reset_index(drop=True),
        "y_reg_train": y_reg.iloc[train_idx].reset_index(drop=True),
        "y_reg_test": y_reg.iloc[test_idx].reset_index(drop=True),
        "y_clf_train": y_clf.iloc[train_idx].reset_index(drop=True),
        "y_clf_test": y_clf.iloc[test_idx].reset_index(drop=True),
    }

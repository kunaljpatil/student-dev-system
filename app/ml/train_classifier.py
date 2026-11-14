"""
Model 2: Student Performance Risk Classification Model (spec Section 6.1, 6.5).

Algorithm: XGBoost Classifier (assigns performance/support category).
Target: 3 classes (On Track, Needs Monitoring, Needs Support).
Baseline: Multinomial Logistic Regression.
Metrics: Accuracy, Macro F1, Precision/Recall per class, Confusion Matrix.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from app.ml.preprocessing import INVERSE_LABEL_MAP


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 2024,
) -> Tuple[XGBClassifier, Pipeline, Dict[str, Any]]:
    """
    Trains the XGBoost Classifier model and the Logistic Regression baseline.
    """
    # 1. XGBoost Classifier (shallow: max_depth 3, learning_rate 0.08)
    xgb_clf = XGBClassifier(
        max_depth=3,
        learning_rate=0.08,
        n_estimators=120,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    xgb_clf.fit(X_train, y_train)

    # 2. Baseline: Logistic Regression (with standard scaling in a pipeline)
    logistic_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    logistic_base.fit(X_train, y_train)

    # Predictions
    y_pred_xgb = xgb_clf.predict(X_test)
    y_pred_log = logistic_base.predict(X_test)

    # Metrics
    acc_xgb = float(accuracy_score(y_test, y_pred_xgb))
    macro_f1_xgb = float(f1_score(y_test, y_pred_xgb, average="macro"))

    acc_log = float(accuracy_score(y_test, y_pred_log))
    macro_f1_log = float(f1_score(y_test, y_pred_log, average="macro"))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_xgb, labels=[0, 1, 2], zero_division=0
    )

    per_class = {}
    for idx, class_name in INVERSE_LABEL_MAP.items():
        per_class[class_name] = {
            "precision": round(float(precision[idx]), 3),
            "recall": round(float(recall[idx]), 3),
            "f1": round(float(f1[idx]), 3),
            "support": int(support[idx]),
        }

    conf_mat = confusion_matrix(y_test, y_pred_xgb, labels=[0, 1, 2]).tolist()

    metrics = {
        "model_name": "Student Performance Risk Classification Model (XGBoost Classifier)",
        "accuracy": round(acc_xgb, 3),
        "macro_f1": round(macro_f1_xgb, 3),
        "per_class": per_class,
        "confusion_matrix": conf_mat,
        "classes": [INVERSE_LABEL_MAP[i] for i in range(3)],
        "baseline": {
            "model_name": "Multinomial Logistic Regression (baseline)",
            "accuracy": round(acc_log, 3),
            "macro_f1": round(macro_f1_log, 3),
        },
    }

    return xgb_clf, logistic_base, metrics

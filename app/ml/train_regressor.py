"""
Model 1: Student Performance Prediction Model (spec Section 6.1, 6.5).

Algorithm: XGBoost Regressor (predicts overall performance score 0-100).
Baseline: Ridge Regression.
Metrics: MAE, RMSE, R2, 5-fold cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 2024,
) -> Tuple[XGBRegressor, Ridge, Dict[str, Any]]:
    """
    Trains the XGBoost Regressor model and the Ridge baseline, computing comparative metrics.
    """
    # 1. XGBoost Regressor (kept shallow as per Section 6.5: max_depth 3-5, learning_rate 0.05-0.1)
    xgb_reg = XGBRegressor(
        max_depth=4,
        learning_rate=0.08,
        n_estimators=150,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_reg.fit(X_train, y_train)

    # 2. Baseline: Ridge Regression
    ridge_base = Ridge(alpha=1.0, random_state=random_state)
    ridge_base.fit(X_train, y_train)

    # Predictions
    y_pred_xgb = xgb_reg.predict(X_test)
    y_pred_ridge = ridge_base.predict(X_test)

    # Metrics on test set
    mae_xgb = float(mean_absolute_error(y_test, y_pred_xgb))
    rmse_xgb = float(root_mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = float(r2_score(y_test, y_pred_xgb))

    mae_ridge = float(mean_absolute_error(y_test, y_pred_ridge))
    rmse_ridge = float(root_mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = float(r2_score(y_test, y_pred_ridge))

    # 5-fold cross validation for XGBoost
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(xgb_reg, X_train, y_train, cv=kfold, scoring="r2")
    cv_r2_mean = float(np.mean(cv_scores))
    cv_r2_std = float(np.std(cv_scores))

    metrics = {
        "model_name": "Student Performance Prediction Model (XGBoost Regressor)",
        "mae": round(mae_xgb, 2),
        "rmse": round(rmse_xgb, 2),
        "r2": round(r2_xgb, 3),
        "cv_r2_mean": round(cv_r2_mean, 3),
        "cv_r2_std": round(cv_r2_std, 3),
        "baseline": {
            "model_name": "Ridge Regression (baseline)",
            "mae": round(mae_ridge, 2),
            "rmse": round(rmse_ridge, 2),
            "r2": round(r2_ridge, 3),
        },
    }

    return xgb_reg, ridge_base, metrics

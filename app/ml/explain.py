"""
Feature Explainability Module with SHAP (spec Section 6.6).

Computes local feature contributions for individual student predictions using
SHAP TreeExplainer on the XGBoost Regressor model.
Extracts top 5 positive and top 5 negative contributors for storage in ml_predictions.shap_json.
Also provides cohort-level global feature importance for the Admin dashboard.
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from xgboost import XGBRegressor

from app.ml.features import FEATURE_NAMES, FEATURE_LABELS


class StudentExplainer:
    """
    Wraps shap.TreeExplainer for the Student Performance Prediction Model (XGBoost Regressor).
    """

    def __init__(self, model: XGBRegressor, feature_names: List[str] = FEATURE_NAMES):
        self.model = model
        self.feature_names = feature_names
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(model)
        # Check base value
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (np.ndarray, list)):
            self.base_value = float(expected_val[0])
        else:
            self.base_value = float(expected_val)

    def explain_student(
        self,
        features: Union[List[float], np.ndarray, pd.DataFrame, pd.Series],
    ) -> Dict[str, Any]:
        """
        Explains an individual student's prediction.
        Returns:
            Dict containing:
            - base_value: expected baseline score
            - predicted_score: model's predicted final score
            - top_positive: top 5 features pushing score higher
            - top_negative: top 5 features lowering score
            - all_factors: full list of 24 features with values & SHAP contributions
        """
        # Ensure 2D DataFrame with proper columns
        if isinstance(features, pd.DataFrame):
            df = features[self.feature_names].copy()
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features.values], columns=self.feature_names)
        elif isinstance(features, (list, np.ndarray)):
            df = pd.DataFrame([features], columns=self.feature_names)
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(df)
        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        elif len(shap_values.shape) > 1:
            sv = shap_values[0]
        else:
            sv = shap_values

        predicted_score = float(self.model.predict(df)[0])
        student_values = df.iloc[0].to_dict()

        factor_list = []
        for feat_name, shap_val in zip(self.feature_names, sv):
            s_val = float(shap_val)
            factor_list.append({
                "feature": feat_name,
                "label": FEATURE_LABELS.get(feat_name, feat_name),
                "value": round(float(student_values[feat_name]), 2),
                "shap_value": round(s_val, 3),
            })

        # Sort positive factors descending
        positive_factors = [f for f in factor_list if f["shap_value"] > 0]
        positive_factors.sort(key=lambda x: x["shap_value"], reverse=True)
        top_positive = positive_factors[:5]

        # Sort negative factors ascending (most negative first)
        negative_factors = [f for f in factor_list if f["shap_value"] < 0]
        negative_factors.sort(key=lambda x: x["shap_value"])
        top_negative = negative_factors[:5]

        return {
            "base_value": round(self.base_value, 2),
            "predicted_score": round(predicted_score, 2),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "all_factors": factor_list,
        }

    def compute_global_importance(self, X_sample: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculates mean absolute SHAP value for each feature across a cohort sample.
        Used for Admin feature importance ranking and charts.
        """
        df = X_sample[self.feature_names].copy()
        shap_values = self.explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance_list = []
        for feat_name, imp in zip(self.feature_names, mean_abs_shap):
            importance_list.append({
                "feature": feat_name,
                "label": FEATURE_LABELS.get(feat_name, feat_name),
                "importance": round(float(imp), 3),
            })

        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        return importance_list

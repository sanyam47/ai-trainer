import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from backend.pipelines.base import ModelPipeline

class NumericRegressionPipeline(ModelPipeline):
    def train(self, filepath: str, intent: dict) -> tuple[str, float, str]:
        """
        Regression Pipeline:
        Expects a CSV with multiple numerical features and one 'target' column.
        """
        if not filepath.endswith(".csv"):
            return None, 0.0, "Only CSV supported for regression"

        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            return None, 0.0, f"CSV read error: {e}"

        if data.empty:
            return None, 0.0, "Dataset is empty"

        # Identify target and features
        # We assume the last column is the target if not specified
        target_col = data.columns[-1]
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Basic cleaning: convert stay only with numeric columns
        X = X.select_dtypes(include=[np.number])
        if X.empty:
            return None, 0.0, "No numeric features found for regression"

        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        if len(X) < 10:
            return None, 0.0, "Need at least 10 samples for regression"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Use Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Use R^2 score for accuracy in context of regression
        score = model.score(X_test, y_test)

        os.makedirs("models", exist_ok=True)
        model_path = "models/regression_model.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, "models/regression_scaler.pkl")

        return model_path, round(score, 3), "Regression training successful"

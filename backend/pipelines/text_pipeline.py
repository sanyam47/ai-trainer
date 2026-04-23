import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from backend.pipelines.base import ModelPipeline

class TextClassificationPipeline(ModelPipeline):
    def train(self, filepath: str, intent: dict) -> tuple[str, float, str]:
        if not filepath.endswith(".csv"):
            return None, 0.0, "Only CSV supported"

        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            return None, 0.0, f"CSV read error: {e}"

        if "text" not in data.columns or "label" not in data.columns:
            return None, 0.0, "CSV must contain 'text' and 'label' columns"

        if data.empty:
            return None, 0.0, "Dataset is empty"

        data = data.dropna()
        data["text"] = data["text"].astype(str)

        X_text = data["text"]
        y = data["label"]

        if len(set(y)) < 2:
            return None, 0.0, "Need at least 2 different labels"

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(X_text)

        # Ensure we can stratify
        class_counts = y.value_counts()
        test_size_val = 0.2
        n_test = int(len(y) * test_size_val)
        n_classes = len(class_counts)
        
        # Extremely safe check: need at least 2 per class AND test set must fit all classes
        can_stratify = (len(y) >= 20) and (class_counts >= 2).all() and (n_test >= n_classes)
        
        print(f"DEBUG: len(y)={len(y)}, n_test={n_test}, n_classes={n_classes}, can_stratify={can_stratify}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_val, random_state=42, 
            stratify=y if can_stratify else None
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        vectorizer_path = "models/vectorizer.pkl"

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        # Also save a model-specific vectorizer so predictions always find the right one
        joblib.dump(vectorizer, "models/model_vectorizer.pkl")

        return model_path, round(accuracy, 3), "Training successful"

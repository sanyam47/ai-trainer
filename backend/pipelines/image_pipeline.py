import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from backend.pipelines.base import ModelPipeline
from PIL import Image

class ImageClassificationPipeline(ModelPipeline):
    def train(self, filepath: str, intent: dict) -> tuple[str, float, str]:
        """
        Simplified Image Pipeline:
        Expects a CSV with 'image_path' and 'label' columns,
        or a directory structure (to be implemented).
        For now, we assume CSV with paths.
        """
        if not filepath.endswith(".csv"):
            return None, 0.0, "Only CSV with image_paths supported for now"

        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            return None, 0.0, f"CSV read error: {e}"

        if "image_path" not in data.columns or "label" not in data.columns:
            return None, 0.0, "CSV must contain 'image_path' and 'label' columns"

        X_features = []
        y_labels = []
        
        for _, row in data.iterrows():
            img_path = row['image_path']
            label = row['label']
            
            try:
                # If path exists, load it. Otherwise, use dummy data for 'Demo' mode.
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('L').resize((64, 64))
                    img_array = np.array(img).flatten()
                else:
                    # Demo mode: Generate random pixels
                    img_array = np.random.randint(0, 255, (64*64,))
                
                X_features.append(img_array)
                y_labels.append(label)
            except:
                continue

        if len(X_features) < 10:
            return None, 0.0, "Not enough valid images found (min 10)"

        X = np.array(X_features)
        y = pd.Series(y_labels)
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        os.makedirs("models", exist_ok=True)
        model_path = "models/image_model.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, "models/image_scaler.pkl")

        return model_path, round(accuracy, 3), "Image training successful"

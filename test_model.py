import joblib
import os

def test_text_model():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        return

    print(f"📦 Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Simple prompt for testing
    test_text = input("\nEnter some text to classify: ")
    
    # Predicting
    prediction = model.predict([test_text])[0]
    print(f"\n🔍 AI Prediction: {prediction}")

def test_regression_model():
    model_path = "models/regression_model.pkl"
    scaler_path = "models/regression_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        return

    print(f"📦 Loading Regression model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    # We assume simple numeric input for now
    val = input("\nEnter a numeric feature (or comma separated features): ")
    try:
        features = [float(x) for x in val.split(",")]
        X = [features]
        if scaler:
            X = scaler.transform(X)
        
        prediction = model.predict(X)[0]
        print(f"\n🔍 AI Predicted Value: {prediction}")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

def test_image_model():
    model_path = "models/image_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        return

    print(f"📦 Loading Image model from {model_path}...")
    model = joblib.load(model_path)
    
    # Needs PIL/Pillow
    from PIL import Image
    import numpy as np

    img_path = input("\nEnter the path to an image file (e.g. dog.jpg): ").strip().strip('"').strip("'")
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    try:
        # Same preprocessing as our Image Pipeline
        img = Image.open(img_path).convert('L').resize((64, 64))
        img_array = np.array(img).flatten().reshape(1, -1)
        
        prediction = model.predict(img_array)[0]
        print(f"\n🔍 AI Prediction: {prediction}")
    except Exception as e:
        print(f"❌ Error processing image: {e}")

if __name__ == "__main__":
    print("--- AI Model Tester ---")
    print("1. Test Text Classification (model.pkl)")
    print("2. Test Regression (regression_model.pkl)")
    print("3. Test Image Classification (image_model.pkl)")
    choice = input("Select a test (1/2/3): ")
    
    if choice == "1":
        test_text_model()
    elif choice == "2":
        test_regression_model()
    elif choice == "3":
        test_image_model()
    else:
        print("Invalid choice.")

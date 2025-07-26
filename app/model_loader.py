import os
import joblib

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    if size_mb > 200:
        raise RuntimeError(f"Model size exceeds 200MB: {size_mb:.2f} MB")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    return model

import joblib
import pandas as pd
import logging

# Initialize Logger
logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads trained ML model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Load model from pickle file."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model successfully loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

    def predict(self, features: list):
        """Predict output based on extracted features."""
        if self.model is None:
            raise ValueError("Model has not been loaded. Call `load()` first.")
        features_df = pd.DataFrame([features])
        return self.model.predict(features_df)[0]

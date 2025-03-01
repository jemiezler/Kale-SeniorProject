import joblib
import pandas as pd
import logging

# Initialize Logger
logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads trained ML model and extracts expected features."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.expected_features = None  # Store expected features

    def load(self):
        """Load model from pickle file and extract expected feature names."""
        try:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, "feature_names_in_"):  # Check if the model stores feature names
                self.expected_features = list(self.model.feature_names_in_)
                logger.info(f"✅ Model loaded successfully! Expected features: {self.expected_features}")
            else:
                logger.warning("⚠️ Model does not have `feature_names_in_`. Using default feature selection.")
                self.expected_features = None  # Handle models that don't store feature names
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

    def predict(self, features: pd.DataFrame):
        """Predict output based on extracted features."""
        if self.model is None:
            raise ValueError("Model has not been loaded. Call `load()` first.")
        return self.model.predict(features)[0]

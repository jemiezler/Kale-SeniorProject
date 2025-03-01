from domain.feature_extractor import FeatureExtractor
from infrastructure.model_loader import ModelLoader
from infrastructure.image_loader import ImageLoader
import pandas as pd

# Load the trained model
model_loader = ModelLoader("models/model.pkl")
model_loader.load()

class AnalysisService:
    """Processes images to extract features and predict."""

    @staticmethod
    def analyze_image(image_bytes: bytes, temp: float):
        """Load image → Extract features → Predict."""
        image = ImageLoader.load(image_bytes)
        features = FeatureExtractor.extract_all_features(image, temp)

        # ✅ Convert to DataFrame with correct column names
        features_df = pd.DataFrame([features])

        # ✅ Keep only the features expected by the model
        filtered_features = features_df[model_loader.expected_features]  # Ensure column order matches training data

        # ✅ Ensure DataFrame format (instead of raw NumPy array)
        prediction = model_loader.predict(filtered_features)  # ✅ Now retains feature names

        return {"prediction": prediction, "features": features}
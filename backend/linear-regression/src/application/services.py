from domain.feature_extractor import FeatureExtractor
from infrastructure.model_loader import ModelLoader
from infrastructure.image_loader import ImageLoader

# Load the trained model
model_loader = ModelLoader("models/model.pkl")
model_loader.load()

class TextureAnalysisService:
    """Processes images to extract features and predict."""

    @staticmethod
    def analyze_texture(image_bytes: bytes):
        """Load image → Extract features → Predict."""
        image = ImageLoader.load(image_bytes)
        features = FeatureExtractor.extract_all_features(image)
        prediction = model_loader.predict(features)

        return {"prediction": prediction, "features": features}

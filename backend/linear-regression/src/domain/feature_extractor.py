import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# Color space mapping
COLOR_SPACES = {
    "RGB": (None, ["R", "G", "B"]),
    "LAB": (cv2.COLOR_BGR2LAB, ["L", "A", "B"]),
    "HSV": (cv2.COLOR_BGR2HSV, ["H", "S", "V"]),
    "GRAY": (cv2.COLOR_BGR2GRAY, ["Gray"])
}

# GLCM Properties
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

class FeatureExtractor:
    """Extracts color and texture features from images."""

    @staticmethod
    def extract_color_features(image: np.ndarray):
        """Extracts mean and std from multiple color spaces."""
        color_features = []
        for space, (conversion, channels) in COLOR_SPACES.items():
            img = cv2.cvtColor(image, conversion) if conversion else image
            mean_vals = np.mean(img, axis=(0, 1)).tolist()
            std_vals = np.std(img, axis=(0, 1)).tolist()
            color_features.extend(mean_vals + std_vals)
        return color_features

    @staticmethod
    def extract_glcm_features(image_gray):
        """Extracts GLCM features from grayscale image."""
        glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        return [graycoprops(glcm, prop).flatten()[0] for prop in GLCM_PROPS]

    @staticmethod
    def extract_lbp_features(image_gray):
        """Extracts Local Binary Pattern (LBP) histogram features."""
        lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        return hist.astype(float).tolist()

    @staticmethod
    def extract_hog_features(image_gray):
        """Extracts Histogram of Oriented Gradients (HOG) features."""
        return hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)[:10]  # Reduce feature size

    @staticmethod
    def extract_all_features(image: np.ndarray):
        """Extracts all features (color + texture) from an image."""
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        color_features = FeatureExtractor.extract_color_features(image)
        glcm_features = FeatureExtractor.extract_glcm_features(image_gray)
        lbp_features = FeatureExtractor.extract_lbp_features(image_gray)
        hog_features = FeatureExtractor.extract_hog_features(image_gray)

        return color_features + glcm_features + lbp_features + hog_features


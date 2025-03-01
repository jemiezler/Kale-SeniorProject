import joblib

# Load model
model = joblib.load("model.pkl")

# Check if model stores feature names
if hasattr(model, "feature_names_in_"):
    print("✅ Model feature names:", model.feature_names_in_)
else:
    print("⚠️ Model does NOT have `feature_names_in_`. You need to provide feature names manually.")

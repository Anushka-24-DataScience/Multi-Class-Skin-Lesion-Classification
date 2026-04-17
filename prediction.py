import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import json


# HAM10000 class labels with full names
CLASS_INFO = {
    "akiec": "Actinic Keratoses",
    "bcc"  : "Basal Cell Carcinoma",
    "bkl"  : "Benign Keratosis",
    "df"   : "Dermatofibroma",
    "mel"  : "Melanoma",
    "nv"   : "Melanocytic Nevi",
    "vasc" : "Vascular Lesions"
}

# Risk level per class — useful for app UI
RISK_LEVEL = {
    "akiec": "High",
    "bcc"  : "High",
    "bkl"  : "Low",
    "df"   : "Low",
    "mel"  : "Very High",
    "nv"   : "Low",
    "vasc" : "Medium"
}


class SkinCancerPredictor:

    def __init__(
        self,
        model_path: str = "artifacts/training/trained_model_local.h5",  # ← use this
        class_indices_path: str = "artifacts/data_transformation/class_indices.json"
    ):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = (380, 380)

        # Load class indices saved during data transformation
        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)

        # Reverse mapping: {0: 'akiec', 1: 'bcc', ...}
        self.idx_to_class = {v: k for k, v in class_indices.items()}

    def preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.img_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)   # (1, 380, 380, 3)
        # No manual rescaling — EfficientNetB4 handles it internally
        return arr

    def predict(self, image_path: str) -> dict:
        arr = self.preprocess(image_path)
        probs = self.model.predict(arr, verbose=0)[0]  # shape (7,)

        pred_idx = int(np.argmax(probs))
        pred_class = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        # All class probabilities
        all_probs = {
            self.idx_to_class[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }

        return {
            "predicted_class"   : pred_class,
            "full_name"         : CLASS_INFO[pred_class],
            "confidence"        : round(confidence, 2),
            "risk_level"        : RISK_LEVEL[pred_class],
            "all_probabilities" : all_probs,
            "recommendation"    : self._get_recommendation(pred_class, confidence)
        }

    def _get_recommendation(self, pred_class: str, confidence: float) -> str:
        if pred_class == "mel" and confidence > 70:
            return "High likelihood of Melanoma detected. Please consult a dermatologist immediately."
        elif pred_class in ["bcc", "akiec"] and confidence > 70:
            return "Potential malignant lesion detected. Medical consultation recommended."
        elif confidence < 50:
            return "Low confidence prediction. Please consult a dermatologist for proper diagnosis."
        else:
            return "Likely benign lesion. Monitor for changes and consult a doctor if concerned."
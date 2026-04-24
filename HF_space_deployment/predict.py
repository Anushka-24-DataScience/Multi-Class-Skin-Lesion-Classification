import numpy as np
import tensorflow as tf
from PIL import Image
import json


CLASS_INFO = {
    "akiec": "Actinic Keratoses",
    "bcc"  : "Basal Cell Carcinoma",
    "bkl"  : "Benign Keratosis",
    "df"   : "Dermatofibroma",
    "mel"  : "Melanoma",
    "nv"   : "Melanocytic Nevi",
    "vasc" : "Vascular Lesions"
}

RISK_LEVEL = {
    "akiec": "🔴 High Risk",
    "bcc"  : "🔴 High Risk",
    "bkl"  : "🟢 Low Risk",
    "df"   : "🟢 Low Risk",
    "mel"  : "🚨 Very High Risk",
    "nv"   : "🟢 Low Risk",
    "vasc" : "🟡 Medium Risk"
}

RECOMMENDATION = {
    "mel"  : "⚠️ High likelihood of Melanoma. Please consult a dermatologist immediately.",
    "bcc"  : "⚠️ Potential Basal Cell Carcinoma. Medical consultation recommended.",
    "akiec": "⚠️ Potential Actinic Keratoses. Medical consultation recommended.",
    "bkl"  : "✅ Likely benign Keratosis. Monitor for changes.",
    "df"   : "✅ Likely Dermatofibroma. Consult a doctor if concerned.",
    "nv"   : "✅ Likely benign Nevus. Monitor for changes.",
    "vasc" : "🟡 Vascular lesion detected. Consult a doctor for evaluation."
}


class SkinCancerPredictor:
    def __init__(
        self,
        # ✅ Use the .h5 file that actually exists on HF
        model_path: str = "artifacts/training/trained_model_hf.h5",
        class_indices_path: str = "artifacts/data_transformation/class_indices.json"
    ):
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = (380, 380)

        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)

        # Reverse mapping: {0: 'akiec', 1: 'bcc', ...}
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        print("✅ Model loaded successfully")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Accepts a PIL Image directly — Gradio passes images as PIL.
        No rescaling — EfficientNetB4 handles preprocessing internally.
        """
        img = image.convert("RGB")
        img = img.resize(self.img_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # (1, 380, 380, 3)
        return arr

    def predict(self, image: Image.Image) -> dict:
        arr = self.preprocess(image)
        probs = self.model.predict(arr, verbose=0)[0]  # shape (7,)

        pred_idx   = int(np.argmax(probs))
        pred_class = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        # All class probabilities for Gradio Label component
        all_probs = {
            CLASS_INFO[self.idx_to_class[i]]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "predicted_class"  : pred_class,
            "full_name"        : CLASS_INFO[pred_class],
            "confidence"       : round(confidence, 2),
            "risk_level"       : RISK_LEVEL[pred_class],
            "recommendation"   : RECOMMENDATION.get(pred_class, "Consult a dermatologist."),
            "all_probabilities": all_probs
        }
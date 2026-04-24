import gradio as gr
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import tempfile
import os

# ── Load model once at startup ──
MODEL_PATH = "artifacts/training/trained_model_hf.h5"
CLASS_INDICES_PATH = "artifacts/data_transformation/class_indices.json"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

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

def predict(image):
    # Preprocess
    img = Image.fromarray(image).convert("RGB")
    img = img.resize((380, 380))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 380, 380, 3)
    # No rescaling — EfficientNetB4 handles internally

    # Predict
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # Format outputs
    label = f"{CLASS_INFO[pred_class]}"
    risk = RISK_LEVEL[pred_class]
    conf_text = f"{confidence:.1f}%"

    # All probabilities for label component
    all_probs = {
        CLASS_INFO[idx_to_class[i]]: float(probs[i])
        for i in range(len(probs))
    }

    # Recommendation
    if pred_class == "mel" and confidence > 60:
        rec = "⚠️ High likelihood of Melanoma. Please consult a dermatologist immediately."
    elif pred_class in ["bcc", "akiec"] and confidence > 60:
        rec = "⚠️ Potential malignant lesion detected. Medical consultation recommended."
    elif confidence < 50:
        rec = "⚠️ Low confidence. Please consult a dermatologist for proper diagnosis."
    else:
        rec = "✅ Likely benign. Monitor for changes and consult a doctor if concerned."

    return label, risk, conf_text, rec, all_probs


# ── Gradio Interface ──
with gr.Blocks(title="DermaCancerScan") as demo:
    gr.Markdown("""
    # 🔬 DermaCancerScan
    ### Skin Cancer Classification using EfficientNetB4
    Trained on HAM10000 dataset — 7 lesion types | AUC: 0.959
    
    > ⚠️ **Disclaimer**: This is a research tool only. 
    Always consult a qualified dermatologist for medical diagnosis.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="Upload Skin Lesion Image"
            )
            submit_btn = gr.Button("🔍 Analyse", variant="primary")

        with gr.Column():
            output_label     = gr.Text(label="Predicted Condition")
            output_risk      = gr.Text(label="Risk Level")
            output_conf      = gr.Text(label="Confidence")
            output_rec       = gr.Text(label="Recommendation")
            output_probs     = gr.Label(
                label="All Class Probabilities",
                num_top_classes=7
            )

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            output_label,
            output_risk,
            output_conf,
            output_rec,
            output_probs
        ]
    )

    gr.Markdown("""
    ### About this model
    - **Architecture**: EfficientNetB4 with custom classification head
    - **Dataset**: HAM10000 (10,015 dermoscopy images)
    - **Classes**: 7 skin lesion types
    - **Metrics**: Accuracy 74.6% | AUC 0.959 | Weighted F1 0.722
    - **Pipeline**: DVC + MLflow for experiment tracking
    """)

# demo.launch()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
from flask import Flask, request, jsonify, render_template
from prediction import SkinCancerPredictor
import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Then access your token
token = os.environ.get("HF_TOKEN")

print(token)  # optional (only for testing)

app = Flask(__name__)
predictor = SkinCancerPredictor()   # loads model once at startup

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        result = predictor.predict(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)   # cleanup


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
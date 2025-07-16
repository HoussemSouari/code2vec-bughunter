import os
import logging
from flask import Flask, request, jsonify
from inference import CodeInference

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the model
model_path = os.environ.get("MODEL_PATH", "models/code2vec_bughunter.pt")
inference_engine = CodeInference(model_path)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "No code provided"}), 400

    code = data["code"]
    try:
        result = inference_engine.predict(code)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return jsonify({"error": "An error occurred during analysis."}), 500

from train import train_model

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    data_path = data.get("data_path", "data/codesearchnet_python")
    model_path = data.get("model_path", "models/code2vec_bughunter.pt")
    epochs = data.get("epochs", 1) # Using 1 epoch for quick testing
    batch_size = data.get("batch_size", 32)
    learning_rate = data.get("learning_rate", 0.001)
    embedding_dim = data.get("embedding_dim", 128)

    try:
        train_model(
            data_path=data_path,
            model_path=model_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
        )
        return jsonify({"message": "Training started."})
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return jsonify({"error": "An error occurred during training."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)

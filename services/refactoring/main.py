import os
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/suggest_refactoring", methods=["POST"])
def suggest_refactoring():
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "No code provided"}), 400

    code = data["code"]
    # TODO: Replace this with a call to an actual LLM
    # from langchain.llms import OpenAI
    # llm = OpenAI()
    # prompt = f"Suggest refactorings for the following code:\n\n{code}"
    # suggestions = llm(prompt)
    suggestions = [
        {
            "line": 3,
            "suggestion": "Placeholder: Extract method to improve readability."
        }
    ]
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)

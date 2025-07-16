import os
import logging
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Service URLs
PROBLEM_DETECTION_URL = os.environ.get("PROBLEM_DETECTION_URL", "http://localhost:5001")
REFACTORING_URL = os.environ.get("REFACTORING_URL", "http://localhost:5002")
TEST_GENERATION_URL = os.environ.get("TEST_GENERATION_URL", "http://localhost:5003")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_code():
    code = request.form.get("code", "")
    if not code:
        return jsonify({"success": False, "error": "No code provided"})

    try:
        # 1. Problem Detection
        response = requests.post(f"{PROBLEM_DETECTION_URL}/analyze", json={"code": code})
        response.raise_for_status()
        problem_detection_result = response.json()

        # 2. Refactoring Suggestions (if problems are found)
        refactoring_suggestions = []
        if problem_detection_result.get("is_buggy"):
            response = requests.post(f"{REFACTORING_URL}/suggest_refactoring", json={"code": code})
            response.raise_for_status()
            refactoring_suggestions = response.json().get("suggestions", [])

        # 3. Test Generation
        response = requests.post(f"{TEST_GENERATION_URL}/generate_tests", json={"code": code})
        response.raise_for_status()
        test_generation_result = response.json()

        return jsonify({
            "success": True,
            "problem_detection": problem_detection_result,
            "refactoring_suggestions": refactoring_suggestions,
            "generated_tests": test_generation_result.get("tests"),
        })

    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with a service: {e}")
        return jsonify({"success": False, "error": "A backend service is unavailable."}), 503
    except Exception as e:
        logging.error(f"Error analyzing code: {e}")
        return jsonify({"success": False, "error": f"An error occurred: {e}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

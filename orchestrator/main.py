import os
import logging
import requests

logging.basicConfig(level=logging.INFO)

# Service URLs
PROBLEM_DETECTION_URL = os.environ.get("PROBLEM_DETECTION_URL", "http://localhost:5001")
REFACTORING_URL = os.environ.get("REFACTORING_URL", "http://localhost:5002")
TEST_GENERATION_URL = os.environ.get("TEST_GENERATION_URL", "http://localhost:5003")

class Orchestrator:
    def __init__(self):
        pass

    def execute_workflow(self, code):
        try:
            # 1. Problem Detection
            response = requests.post(f"{PROBLEM_DETECTION_URL}/analyze", json={"code": code})
            response.raise_for_status()
            problem_detection_result = response.json()

            # 2. Refactoring Suggestions
            refactoring_suggestions = []
            if problem_detection_result.get("is_buggy"):
                response = requests.post(f"{REFACTORING_URL}/suggest_refactoring", json={"code": code})
                response.raise_for_status()
                refactoring_suggestions = response.json().get("suggestions", [])

            # 3. Test Generation
            response = requests.post(f"{TEST_GENERATION_URL}/generate_tests", json={"code": code})
            response.raise_for_status()
            test_generation_result = response.json()

            return {
                "success": True,
                "problem_detection": problem_detection_result,
                "refactoring_suggestions": refactoring_suggestions,
                "generated_tests": test_generation_result.get("tests"),
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"Error communicating with a service: {e}")
            return {"success": False, "error": "A backend service is unavailable."}
        except Exception as e:
            logging.error(f"Error executing workflow: {e}")
            return {"success": False, "error": f"An error occurred: {e}"}

if __name__ == "__main__":
    orchestrator = Orchestrator()
    sample_code = """
def my_function(x):
    return x + 1
"""
    result = orchestrator.execute_workflow(sample_code)
    import json
    print(json.dumps(result, indent=2))

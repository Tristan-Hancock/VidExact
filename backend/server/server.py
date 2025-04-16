import os
import subprocess
import json
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Allow CORS for all endpoints under /api/ from any origin
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# BASE_DIR points to the 'backend' folder (since __file__ is in backend/server)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("BASE_DIR set to:", BASE_DIR)

# Configure upload folder inside backend/input
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'input')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder at: {UPLOAD_FOLDER}")

ALLOWED_EXTENSIONS = {'mp4', 'webm'}

def allowed_file(filename):
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    print(f"Checking if '{filename}' is allowed: {is_allowed}")
    return is_allowed

@app.route('/api/upload', methods=['POST'])
@cross_origin()
def upload_video():
    print("Received file upload request.")
    if 'file' not in request.files:
        print("No file part in request.")
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Force a standard name (Video4.mp4) so that videxact.py can pick it up.
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Video4.mp4')
        file.save(save_path)
        print(f"File '{filename}' saved to: {save_path}")
        return jsonify({'message': 'File uploaded successfully', 'path': save_path}), 200
    else:
        print(f"File '{file.filename}' is not an allowed file type.")
        return jsonify({'error': 'Invalid file type'}), 400

# ---------------------------------------------------------
# Endpoint to run processing scripts (Phase 1)
# ---------------------------------------------------------
@app.route('/api/process', methods=['POST', 'OPTIONS'])
@cross_origin()
def run_processing():
    """
    This endpoint sequentially runs:
      python videxact.py
      python clean_csv.py
      python nlp_model.py
    All paths are computed based on BASE_DIR.
    """
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    try:
        # Compute absolute paths to the scripts
        videxact_path = os.path.join(BASE_DIR, "app", "videxact.py")
        clean_csv_path = os.path.join(BASE_DIR, "scripts", "clean_csv.py")
        nlp_model_path = os.path.join(BASE_DIR, "scripts", "nlp_model.py")
        
        print("========== Processing Pipeline ==========")
        
        # Step 1: Run videxact.py
        print("Checking if videxact.py exists at:", videxact_path)
        if not os.path.exists(videxact_path):
            print("Error: videxact.py does not exist!")
        else:
            print("========== Starting videxact.py ==========")
            result_videxact = subprocess.run(
                ["python", videxact_path],
                check=True,
                capture_output=True,
                text=True
            )
            print("---------- videxact.py Output ----------")
            print(result_videxact.stdout)
            print("========== videxact.py Completed ==========")
        
        # Step 2: Run clean_csv.py
        print("Checking if clean_csv.py exists at:", clean_csv_path)
        if not os.path.exists(clean_csv_path):
            print("Error: clean_csv.py does not exist!")
        else:
            print("========== Starting clean_csv.py ==========")
            result_clean_csv = subprocess.run(
                ["python", clean_csv_path],
                check=True,
                capture_output=True,
                text=True
            )
            print("---------- clean_csv.py Output ----------")
            print(result_clean_csv.stdout)
            print("========== clean_csv.py Completed ==========")
        
        # Step 3: Run nlp_model.py
        print("Checking if nlp_model.py exists at:", nlp_model_path)
        if not os.path.exists(nlp_model_path):
            print("Error: nlp_model.py does not exist!")
        else:
            print("========== Starting nlp_model.py ==========")
            result_nlp_model = subprocess.run(
                ["python", nlp_model_path],
                check=True,
                capture_output=True,
                text=True
            )
            print("---------- nlp_model.py Output ----------")
            print(result_nlp_model.stdout)
            print("========== nlp_model.py Completed ==========")
        
        return jsonify({"message": "Processing complete"}), 200

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print("Error during processing:", error_msg)
        return jsonify({"error": "Processing failed", "details": error_msg}), 500

# ---------------------------------------------------------
# Endpoint for search (Phase 2)
# ---------------------------------------------------------
@app.route('/api/search', methods=['POST', 'OPTIONS'])
@cross_origin()
def run_search():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    top_k = data.get("top_k", 5)
    try:
        nlp_search_path = os.path.join(BASE_DIR, "scripts", "nlp_search.py")
        print("========== Starting nlp_search.py ==========")
        print(f"Using path: {nlp_search_path} with query='{query}' and top_k={top_k}")
        proc = subprocess.run(
            [
                "python",
                nlp_search_path,
                query,
                "--top_k",
                str(top_k)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("---------- nlp_search.py Raw Output ----------")
        print(proc.stdout)
        
        # Parse the JSON output from nlp_search.py
        results_data = json.loads(proc.stdout)
        print("========== nlp_search.py Completed ==========")
        return jsonify({"message": "Search complete", "results": results_data["results"]}), 200

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print("Error during NLP search:", error_msg)
        return jsonify({"error": "Search failed", "details": error_msg}), 500
    except json.JSONDecodeError as je:
        print("JSON decode error:", je)
        return jsonify({"error": "Search failed", "details": "Invalid JSON output from nlp_search.py"}), 500



if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=8000, debug=True)

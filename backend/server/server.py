# backend/server/app.py
import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import Flask-CORS


app = Flask(__name__)
CORS(app)
# Set up the upload folder; use an absolute path so it’s not ambiguous.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'input')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder at: {UPLOAD_FOLDER}")

ALLOWED_EXTENSIONS = {'mp4', 'webm'}

def allowed_file(filename):
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    print(f"Checking if '{filename}' is allowed: {is_allowed}")
    return is_allowed

@app.route('/api/upload', methods=['POST'])
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
        # For simplicity, save the file as Video4.mp4 (or you could keep the original filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Video4.mp4')
        file.save(save_path)
        print(f"File '{filename}' saved to: {save_path}")
        return jsonify({'message': 'File uploaded successfully', 'path': save_path}), 200
    else:
        print(f"File '{file.filename}' is not an allowed file type.")
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    print("Starting Flask server for file uploads...")
    # Run on port 8000 and bind to all interfaces.
    app.run(host='0.0.0.0', port=8000, debug=True)

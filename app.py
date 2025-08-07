# app.py (Corrected Version)

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# First, load the secret keys from the .env file.
load_dotenv()

# Second, NOW that the keys are loaded, import the agent file that needs them.
from agent import process_analysis_request

# Third, create the Flask application.
app = Flask(__name__)

# --- Your API route definitions start here ---
@app.route('/api/', methods=['POST'])
def handle_analysis_request():
    if 'questions.txt' not in request.files:
        return jsonify({"error": "questions.txt is missing"}), 400

    questions_file = request.files['questions.txt']
    task_description = questions_file.read().decode('utf-8')
    other_files = {filename: file for filename, file in request.files.items() if filename != 'questions.txt'}

    try:
        result = process_analysis_request(task_description, other_files)
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred. Check server logs."}), 500

# --- This runs the app ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
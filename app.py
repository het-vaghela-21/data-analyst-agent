# app.py
import os
import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# NOW import the agent
from agent import process_analysis_request

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def handle_analysis_request():
    if 'questions.txt' not in request.files:
        return jsonify({"error": "questions.txt is missing"}), 400

    try:
        questions_file = request.files['questions.txt']
        task_description = questions_file.read().decode('utf-8')
        other_files = {filename: file for filename, file in request.files.items() if filename != 'questions.txt'}

        result = process_analysis_request(task_description, other_files)
        return jsonify(result)
    except Exception as e:
        print(f"A critical error occurred in Flask handler: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
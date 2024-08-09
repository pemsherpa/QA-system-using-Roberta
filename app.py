from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline
import os

app = Flask(__name__)

# Load the question answering model
model_path = '/Users/pemasherpa/Desktop/1st year/Project/question-answering/Server/model'
nlp = pipeline('question-answering', model=model_path, tokenizer=model_path)

@app.route('/')
def home():
    return send_from_directory('../client', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../client', path)

@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.json
    question = data.get('question')
    context = data.get('context')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not context:
        return jsonify({"error": "Context is required"}), 400

    # Prepare input for the QA model
    QA_input = {
        'question': question,
        'context': context
    }

    result = nlp(QA_input)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


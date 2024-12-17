from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)

# Load the T5 model and tokenizer with use_fast=True to ensure the fast tokenizer is used
tokenizer = AutoTokenizer.from_pretrained('trained_model', use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained('trained_model')

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/response', methods=['POST'])
def predict():
    data = request.json
    print("Received JSON data:", data)
    
    # Extract the question from the input data
    input_text = data['input']  # assuming input is a string with the question
    
    # Tokenize the input question
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    
    # Generate the response using the model
    with torch.no_grad():
        # Generate the output for the input question
        outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)
    
    # Decode the generated sequence to get the answer
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Prepare the response with the generated answer
    response = {
        'response': decoded_output,
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

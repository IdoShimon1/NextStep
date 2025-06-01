"""
Flask server for resume title classification
Requirements:
pip install flask flask-cors torch transformers pdfminer.six python-docx requests
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import json
import time
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web app communication

# Model state
current_model = None
current_tokenizer = None
current_label_encoder = None
current_model_name = None
MAX_LEN = 256

def find_experiment_directories():
    """Find all experiment directories with models."""
    experiment_dirs = []
    
    # Look specifically in the requested experiment directory
    target_exp_dir = "experiments_20250519_175233"
    if os.path.exists(target_exp_dir) and os.path.isdir(target_exp_dir):
        logger.info(f"Looking for models in: {target_exp_dir}")
        
        # Each model directory is directly inside the experiment directory
        for model_dir in sorted(glob.glob(f"{target_exp_dir}/*/"), reverse=True):
            model_dir = model_dir.rstrip("/\\")
            model_name = os.path.basename(model_dir).replace("_", " ")
            
            # Check if this is a model directory by looking for checkpoints
            checkpoints = glob.glob(f"{model_dir}/checkpoint-*/")
            if checkpoints:
                # Use the latest checkpoint as the model path
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(os.path.basename(x.rstrip('/\\')).split('-')[1]))[-1].rstrip("/\\")
                
                logger.info(f"Found model: {model_name} at {latest_checkpoint}")
                experiment_dirs.append((model_name, latest_checkpoint))
    
    if not experiment_dirs:
        logger.warning("⚠️  No models found in the specified directory.")
    
    return experiment_dirs

def load_model(model_path, model_name=None):
    """Load a model, tokenizer and label-encoder that BELONG together."""
    global current_model, current_tokenizer, current_label_encoder, current_model_name

    try:
        # ------------ paths --------------------------------------------------
        model_dir      = os.path.dirname(model_path)          # …/model_name
        tokenizer_path = os.path.join(model_dir, "tokenizer") # …/model_name/tokenizer
        encoder_path   = os.path.join(model_dir, "label_encoder.pkl")

        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer dir not found: {tokenizer_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

        # ------------ load objects -------------------------------------------
        logger.info(f"Loading tokenizer  → {tokenizer_path}")
        current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info(f"Loading model      → {model_path}")
        current_model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        current_model.eval()

        logger.info(f"Loading encoder    → {encoder_path}")
        with open(encoder_path, "rb") as fp:
            current_label_encoder = pickle.load(fp)

        current_model_name = model_name or os.path.basename(model_dir).replace("_", " ")
        logger.info(f"✓ Model loaded successfully: {current_model_name}")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def predict_title(resume_json: dict) -> tuple:
    """Predict job title from resume JSON data."""
    if current_model is None or current_tokenizer is None or current_label_encoder is None:
        raise ValueError("Model not loaded")
    
    # Process the resume data exactly as done during training
    text_parts = []
    for ed in resume_json.get("education", []):
        text_parts.append(f"{ed.get('degree','')} {ed.get('field','')} {ed.get('institution','')}")
    
    # Add job history (skip current job)
    jobs = resume_json.get("job_history", [])
    if jobs and jobs[0].get("end_date") is None:
        jobs = jobs[1:]
    for job in jobs:
        text_parts.append(f"{job.get('title','')} {job.get('company','')}")
    
    text_parts.append(" ".join(resume_json.get("skills", [])))
    full_text = " ".join(text_parts)

    enc = current_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = current_model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Handle case where model outputs more logits than we have classes
    num_classes = len(current_label_encoder.classes_)
    if len(probs) > num_classes:
        probs = probs[:num_classes]
    
    predicted_label = current_label_encoder.inverse_transform([probs.argmax()])[0]
    
    # Get top 3 predictions with probabilities
    top_indices = probs.argsort()[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        label = current_label_encoder.inverse_transform([idx])[0]
        # Convert numpy float to Python float and round to 2 decimal places
        probability = float(f"{probs[idx] * 100:.2f}")
        top_predictions.append({
            "title": str(label),  # Ensure label is a string
            "confidence": probability
        })
    
    return str(predicted_label), top_predictions

# Load available models
AVAILABLE_MODELS = find_experiment_directories()

# Find RoBERTa model to use as default
DEFAULT_MODEL = None
for name, path in AVAILABLE_MODELS:
    if "roberta" in name.lower():
        DEFAULT_MODEL = (name, path)
        break

# If no RoBERTa found, use first available model
if not DEFAULT_MODEL and AVAILABLE_MODELS:
    DEFAULT_MODEL = AVAILABLE_MODELS[0]

# Load the default model
if DEFAULT_MODEL:
    if not load_model(DEFAULT_MODEL[1], DEFAULT_MODEL[0]):
        logger.error("Failed to load default model")
else:
    logger.warning("No models available")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if current_model is not None else "not loaded"
    return jsonify({
        "status": "healthy",
        "message": "Model server is running",
        "model_status": model_status,
        "current_model": current_model_name
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    models = [{"name": name, "path": path} for name, path in AVAILABLE_MODELS]
    return jsonify({
        "available_models": models,
        "current_model": current_model_name,
        "default_model": DEFAULT_MODEL[0] if DEFAULT_MODEL else None
    })

@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({"error": "No model name provided"}), 400
        
        model_name = data['model_name']
        model_path = None
        
        # Find the model path for the given name
        for name, path in AVAILABLE_MODELS:
            if name == model_name:
                model_path = path
                break
        
        if not model_path:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        
        # Load the new model
        if load_model(model_path, model_name):
            return jsonify({
                "message": f"Successfully switched to model: {model_name}",
                "current_model": current_model_name
            })
        else:
            return jsonify({"error": f"Failed to load model: {model_name}"}), 500
            
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return jsonify({"error": f"Failed to switch model: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'resume_json' not in data:
            return jsonify({"error": "No resume JSON provided"}), 400
        
        resume_json = data['resume_json']
        
        logger.info(f"Received prediction request for resume")
        
        if current_model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        predicted_title, top_predictions = predict_title(resume_json)
        
        response = {
            "predicted_title": predicted_title,
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "model_used": current_model_name
        }
        
        logger.info(f"Prediction result: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting model server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)

#!/usr/bin/env python3
"""
Test Verification Script
=======================
This script verifies that the test environment is working correctly.
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

def main():
    parser = argparse.ArgumentParser(description="Verify test environment")
    parser.add_argument("--model", default="experiments_20250512_012210/BERT_Base_(higher_lr)/checkpoint-512", 
                      help="Path to model directory")
    parser.add_argument("--resume", default=None, help="Path to resume file")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Find label encoder
    label_path = None
    potential_paths = [
        os.path.join(args.model, "label_encoder.pkl"),
        os.path.join(os.path.dirname(args.model), "label_encoder.pkl"),
        os.path.join(os.path.dirname(os.path.dirname(args.model)), "label_encoder.pkl"),
        "label_encoder.pkl"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            label_path = path
            break
    
    if not label_path:
        print("Could not find label_encoder.pkl")
        sys.exit(1)
    
    print(f"Loading label encoder from {label_path}...")
    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Create sample resume JSON
    resume_json = {
        "uid": "12345",
        "label": "",
        "education": [
            {
                "degree": "Bachelor's Degree",
                "field": "Computer Science",
                "institution": "Example University",
                "year_completed": "2020"
            }
        ],
        "job_history": [
            {
                "job_id": "001",
                "title": "Software Engineer",
                "company": "Tech Company",
                "start_date": "2020-01-01",
                "end_date": None,
                "skills": ["Python", "JavaScript"]
            },
            {
                "job_id": "002",
                "title": "Junior Developer",
                "company": "Startup Inc",
                "start_date": "2018-01-01",
                "end_date": "2019-12-31",
                "skills": ["HTML", "CSS"]
            }
        ],
        "skills": ["Python", "JavaScript", "HTML", "CSS", "SQL", "Git"]
    }
    
    # Process resume data
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
    
    print(f"\nProcessed text: {full_text}")
    
    # Make prediction
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**enc)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Handle case where model outputs more logits than we have classes
    num_classes = len(label_encoder.classes_)
    if len(probs) > num_classes:
        print(f"Model outputs {len(probs)} logits but label encoder has {num_classes} classes")
        probs = probs[:num_classes]
    
    predicted_label = label_encoder.inverse_transform([probs.argmax()])[0]
    
    # Get top predictions
    top_indices = probs.argsort()[-5:][::-1]
    
    print("\nTop predictions:")
    for idx in top_indices:
        label = label_encoder.inverse_transform([idx])[0]
        probability = probs[idx] * 100
        print(f"{label}: {probability:.2f}%")
    
    print(f"\nPredicted title: {predicted_label}")

if __name__ == "__main__":
    main()

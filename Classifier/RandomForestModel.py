import os
import json
import time
import re
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

##############################################
# 1. Configuration & Setup
##############################################

# OpenAI / LLM API Key and URL
API_KEY = ""
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# File Paths:
LABELED_RESUMES_CSV = "Resume_Data/clusters_summary1.csv"  # Labeled resume-to-cluster data
RESUME_TEXT_PATH = "Resume_Data/Ido_Resume.txt"            # New resume to classify

# (Optional) Initialize an embedding model – kept here from your base code
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

##############################################
# 2. LLM Prompt: Extracting Resume Data
##############################################

def extract_info_from_resume(resume_text):
    """
    Calls an LLM (e.g., OpenAI GPT) to parse the resume text and return structured JSON:
      {
        "job_titles": [...],
        "skills": [...],
        "education": "Bachelor's Degree in X",
        "years_of_experience": 4
      }
    For this pipeline, we use the skills, education, and years_of_experience.
    
    This version strips the response and attempts to extract the JSON content if extra text is present.
    """
    system_prompt = "You are a resume parser. Extract structured information from the text and return only valid JSON."
    user_prompt = f"""
    Here is a resume text:

    {resume_text}

    1. List out all professional job titles (in English).
    2. List out the skills mentioned.
    3. List out the highest level of education found (e.g., Bachelor's Degree in X).
    4. Estimate total years of relevant professional experience.

    Return the result in strict JSON format with no additional text. Use the keys exactly:
    "job_titles", "skills", "education", "years_of_experience".
    Example:
    {{
      "job_titles": ["Operations Manager", "Purchasing Agent"],
      "skills": ["Supply Chain", "Logistics", "Inventory Management"],
      "education": "Bachelor's Degree in Supply Chain Management",
      "years_of_experience": 5
    }}
    """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",  # Using your working model name
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        raw_content = response_json["choices"][0]["message"]["content"]
        
        # Clean the response: strip whitespace
        raw_content = raw_content.strip()
        
        # If the response does not start with '{', attempt to extract the JSON portion
        if not raw_content.startswith("{"):
            json_match = re.search(r'({.*})', raw_content, re.DOTALL)
            if json_match:
                raw_content = json_match.group(1)
        
        # Parse JSON
        extracted_data = json.loads(raw_content)
        # Ensure all mandatory keys are present
        for key in ["job_titles", "skills", "education", "years_of_experience"]:
            if key not in extracted_data:
                extracted_data[key] = [] if key in ["job_titles", "skills"] else ""
        return extracted_data
    except Exception as e:
        print("Error calling LLM or parsing JSON:", e)
        print("Raw response:", raw_content)
        return {
            "job_titles": [],
            "skills": [],
            "education": "",
            "years_of_experience": 0
        }

##############################################
# 3. Training a Random Forest Classifier for Clusters
##############################################

def train_random_forest_classifier():
    """
    Loads the labeled resume-to-cluster data from LABELED_RESUMES_CSV,
    builds a feature pipeline using TF-IDF for skills, one-hot encoding for Education,
    and scaling for Experience, and trains a RandomForestClassifier to predict cluster.
    Assumes the CSV has columns:
       - "cluster"
       - "most_common_skills"
       - "most_common_education"
       - "average_experience"
    """
    # Load labeled data
    df = pd.read_csv(LABELED_RESUMES_CSV)
    X = df[["most_common_skills", "most_common_education", "average_experience"]]
    y = df["cluster"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("skills_text", TfidfVectorizer(), "most_common_skills"),
            ("edu_cat", OneHotEncoder(handle_unknown="ignore"), ["most_common_education"]),
            ("num", StandardScaler(), ["average_experience"])
        ]
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

##############################################
# 4. Main Pipeline
##############################################

def main():
    # A) Train the Random Forest classifier using the labeled resume-to-cluster dataset
    clf_pipeline = train_random_forest_classifier()
    
    # B) Read the new resume text from file
    with open(RESUME_TEXT_PATH, "r", encoding="utf-8") as f:
        resume_text = f.read()
    
    # C) Extract structured resume info using the LLM
    resume_data = extract_info_from_resume(resume_text)
    print("Extracted Resume Data:", resume_data)
    
    # D) Prepare features for prediction:
    # Join the skills list into a single string
    skills_str = ", ".join(resume_data.get("skills", []))
    education = resume_data.get("education", "")
    experience = resume_data.get("years_of_experience", 0)
    
    # Create a DataFrame with the same columns as the training data
    new_sample = pd.DataFrame([{
        "most_common_skills": skills_str,
        "most_common_education": education,
        "average_experience": experience
    }])
    
    # E) Predict the cluster for the new resume
    predicted_cluster = clf_pipeline.predict(new_sample)[0]
    print("Predicted cluster for the new resume:", predicted_cluster)
    
if __name__ == "__main__":
    main()

import os
import json
import re
import requests
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression

##############################################
# 1. Configuration & Setup
##############################################

# OpenAI / LLM API Key and URL
API_KEY = ""
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# File Paths:
SYNTHETIC_CSV = "Data/Random_Resumes.csv"   # CSV with synthetic resume-job pairs and computed features
JOBS_CSV = "Data/Enriched_and_cleaned_data.csv"  # CSV with individual job titles (columns: title, skills, Education, Experience)
RESUME_TEXT_PATH = "Data/Ido_Resume.txt"  # Text file for the resume to test
OUTPUT_CSV = "Data/predicted_matches.csv"  # Output CSV for predicted match percentages
MODEL_LR_PATH = "Data/linear_regression_model.pkl"  # File path to save/load the regression model

# Embedding model configuration:
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Maximum experience for normalization (must match your synthetic data generation):
MAX_EXPERIENCE = 15

##############################################
# 2. Education Mapping & Utility Functions
##############################################

education_mapping = {
    "high": 0, 
    "certified": 1, 
    "culinary": 1, 
    "commercial": 1, 
    "diploma": 1, 
    "certificate": 1,
    "associate": 1, 
    "associate's": 1,
    "bachelor": 2,
    "bachelor's": 2,
    "master's": 3,
    "mba": 3,
    "phd": 3,
    "doctoral": 3,
    "ph.d.": 3,
    "doctorate": 3,
    "doctor": 3,
    "juris": 3
}

def extract_skills(skills_str):
    """Convert a comma-separated string into a list of cleaned, lowercase skills."""
    if not isinstance(skills_str, str):
        return []
    return [skill.strip().lower() for skill in skills_str.split(',') if skill.strip()]

def education_level(edu_str):
    """Return a numeric education level based on education_mapping."""
    edu_str = edu_str.lower()
    level = -1
    for key, value in education_mapping.items():
        if key in edu_str:
            level = max(level, value)
    return level if level >= 0 else 0

def meets_or_exceeds(resume_edu, job_edu):
    """Return 1 if the resume's education level meets or exceeds the job's requirement, else 0."""
    return 1 if education_level(resume_edu) >= education_level(job_edu) else 0

def compute_skills_similarity(skills_list1, skills_list2, model):
    """
    Compute the semantic similarity between two lists of skills using BERT embeddings.
    Returns a value between 0 and 1.
    """
    if not skills_list1 or not skills_list2:
        return 0.0
    embeddings1 = model.encode(skills_list1, convert_to_numpy=True)
    embeddings2 = model.encode(skills_list2, convert_to_numpy=True)
    mean1 = np.mean(embeddings1, axis=0)
    mean2 = np.mean(embeddings2, axis=0)
    dot_product = np.dot(mean1, mean2)
    norm_product = np.linalg.norm(mean1) * np.linalg.norm(mean2)
    similarity = dot_product / norm_product if norm_product != 0 else 0.0
    return (similarity + 1) / 2  # Map cosine similarity from [-1,1] to [0,1]

##############################################
# 3. LLM Resume Extraction Function
##############################################

def extract_info_from_resume(resume_text):
    """
    Calls the OpenAI API to extract structured resume information.
    Returns a dict with keys: "job_titles", "skills", "education", "years_of_experience".
    """
    system_prompt = "You are a resume parser. Extract structured information from the text and return only valid JSON."
    user_prompt = f"""
Here is a resume text:

{resume_text}

1. List all professional job titles (in English).
2. List all the skills mentioned.
3. Identify the highest level of education (e.g., Bachelor's Degree in X).
4. Estimate total years of relevant professional experience.

Return the result in strict JSON format with keys exactly:
"job_titles", "skills", "education", "years_of_experience".
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",  # Adjust if needed
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
        raw_content = response_json["choices"][0]["message"]["content"].strip()
        if not raw_content.startswith("{"):
            match = re.search(r'({.*})', raw_content, re.DOTALL)
            if match:
                raw_content = match.group(1)
        extracted = json.loads(raw_content)
        for key in ["job_titles", "skills", "education", "years_of_experience"]:
            if key not in extracted:
                extracted[key] = [] if key in ["job_titles", "skills"] else ""
        return extracted
    except Exception as e:
        print("Error extracting resume info:", e)
        return {"job_titles": [], "skills": [], "education": "", "years_of_experience": 0}

##############################################
# 4. Training Linear Regression Model using Synthetic Data
##############################################

def train_regression_model(synthetic_csv_path, max_experience):
    """
    Loads synthetic data CSV and trains a linear regression model to predict match_score.
    The CSV is expected to have columns: skills_similarity, education_match, experience_diff, match_score.
    We derive experience_closeness = 1 - (experience_diff / max_experience).
    Returns the trained model.
    """
    df = pd.read_csv(synthetic_csv_path)
    df['experience_closeness'] = 1 - (df['experience_diff'] / max_experience)
    features = ['skills_similarity', 'education_match', 'experience_closeness']
    X = df[features]
    y = df['match_score']
    model_lr = LinearRegression()
    model_lr.fit(X, y)
    print("Trained Linear Regression Model:")
    print("Intercept:", model_lr.intercept_)
    print("Coefficients:", dict(zip(features, model_lr.coef_)))
    return model_lr

def get_regression_model(synthetic_csv_path, max_experience, model_path):
    """
    Loads the regression model from disk if available; otherwise trains it and saves it.
    """
    if os.path.exists(model_path):
        print("Loading regression model from disk...")
        with open(model_path, "rb") as f:
            model_lr = pickle.load(f)
    else:
        print("Training regression model from synthetic data...")
        model_lr = train_regression_model(synthetic_csv_path, max_experience)
        with open(model_path, "wb") as f:
            pickle.dump(model_lr, f)
        print(f"Regression model saved to {model_path}")
    return model_lr

##############################################
# 5. Predicting Match Percentage for Each Job Using the Trained Model
##############################################

def predict_match_for_jobs(resume_info, jobs_df, embedder, model_lr, max_experience):
    """
    For each job in jobs_df, compute features (skills similarity, education match,
    experience closeness) based on resume_info and predict the match percentage using model_lr.
    Returns a DataFrame with job titles and predicted match percentages.
    """
    results = []
    for idx, row in jobs_df.iterrows():
        # Skills similarity:
        resume_skills = [s.lower() for s in resume_info.get("skills", [])]
        job_skills = extract_skills(row.get("skills", ""))
        skills_similarity = compute_skills_similarity(job_skills, resume_skills, embedder)
        
        # Education match:
        resume_edu = resume_info.get("education", "")
        job_edu = row.get("Education", "")
        edu_match = meets_or_exceeds(resume_edu, job_edu)
        
        # Experience closeness:
        try:
            resume_exp = float(resume_info.get("years_of_experience", 0))
        except:
            resume_exp = 0
        try:
            job_exp = float(row.get("Experience", 0))
        except:
            job_exp = 0
        experience_diff = abs(resume_exp - job_exp)
        experience_closeness = max(0, 1 - (experience_diff / max_experience))
        
        # Create feature vector and predict match score:
        features = np.array([[skills_similarity, edu_match, experience_closeness]])
        predicted_match = model_lr.predict(features)[0]
        
        results.append({
            "job_title": row.get("title", ""),
            "predicted_match_percentage": predicted_match,
            "skills_similarity": skills_similarity,
            "education_match": edu_match,
            "experience_closeness": experience_closeness,
            "job_Education": row.get("Education", ""),
            "job_Experience": row.get("Experience", "")
        })
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="predicted_match_percentage", ascending=False, inplace=True)
    return results_df

##############################################
# 6. Main Function: Training & Testing Pipeline
##############################################

def main():
    # ----- Step 1: Read and extract resume information via LLM -----
    try:
        with open(RESUME_TEXT_PATH, "r", encoding="utf-8") as f:
            resume_text = f.read()
    except Exception as e:
        print("Error reading resume file:", e)
        return
    print("Extracting resume information via OpenAI...")
    resume_info = extract_info_from_resume(resume_text)
    print("Extracted Resume Info:", resume_info)
    
    # ----- Step 2: (Optional) Use your Random Forest classifier to predict cluster -----
    # (Assume you have already computed clusters and saved them in CLUSTERED_JOBS_CSV.)
    # For this example, we'll simply load the precomputed job titles without filtering by cluster.
    try:
        jobs_df = pd.read_csv(JOBS_CSV)
    except Exception as e:
        print("Error reading jobs CSV:", e)
        return
    
    # ----- Step 3: Train (or load) the Linear Regression model using synthetic data -----
    model_lr = get_regression_model(SYNTHETIC_CSV, MAX_EXPERIENCE, MODEL_LR_PATH)
    
    # ----- Step 4: Predict matching scores for each job using the trained regression model -----
    results_df = predict_match_for_jobs(resume_info, jobs_df, embedder, model_lr, MAX_EXPERIENCE)
    
    # ----- Step 5: Save and print the results -----
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("\n=== Predicted Match Percentages for Each Job ===")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

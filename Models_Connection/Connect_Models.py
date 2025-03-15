import os
import json
import re
import requests
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

##############################################
# CONFIGURATION & FILE PATHS
##############################################

# OpenAI API settings (update API_KEY with your own key)
API_KEY = ""
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# File Paths – update these as needed:
CLUSTERS_SUMMARY_CSV = "Data/clusters_summary1.csv"    # Aggregated cluster summary (must include "jobs")
ENRICHED_DATA_CSV    = "Data/Enriched_and_cleaned_data.csv"  # Enriched job data (with columns: title, skills, Education, Experience)
MODEL_RF_PATH        = "Data/random_forest_classifier.pkl"    # Saved RandomForest classifier for cluster prediction
MODEL_LR_PATH        = "Data/linear_regression_model.pkl"  # Saved Linear Regression model for match scoring
RESUME_TEXT_PATH     = "Resumes/Ido_Resume.txt"             # Resume text to test
SYNTHETIC_CSV        = "Data/Random_Resumes.csv"
OUTPUT_CSV           = "Data/final_predicted_matches.csv"  # Final output CSV with job-level match percentages

# Flags to force retraining if needed:
FORCE_RF_RETRAIN = False
FORCE_LR_RETRAIN = False

# Embedding model configuration:
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Maximum experience for normalization (must match synthetic data generation)
MAX_EXPERIENCE = 15

##############################################
# EDUCATION MAPPING & UTILITY FUNCTIONS
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
    "b.sc": 2,
    "bsc" :2,
    "master's": 3,
    "mba": 3,
    "phd": 3,
    "doctoral": 3,
    "ph.d.": 3,
    "phd" :3,
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
    Compute semantic similarity between two lists of skills using BERT embeddings.
    Returns a value between 0 and 1.
    """
    if not skills_list1 or not skills_list2:
        print("Warning: One or both skill lists are empty. Returning similarity = 0.0")
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
# LLM RESUME EXTRACTION FUNCTION
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

def validate_resume_info(resume_info):
    """Print warnings if key fields in resume_info are empty."""
    if not resume_info.get("skills"):
        print("Warning: No skills were extracted from the resume!")
    if not resume_info.get("education"):
        print("Warning: No education info was extracted from the resume!")

##############################################
# LOAD SAVED MODELS OR TRAIN IF NOT PRESENT
##############################################

def get_classifier_model(model_path):
    """
    Loads the Random Forest classifier from disk if available; else returns an error.
    """
    if os.path.exists(model_path):
        print("Loading Random Forest classifier from disk...")
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        return classifier
    else:
        raise FileNotFoundError(f"Random Forest classifier model not found at {model_path}!")


def get_regression_model(model_path):
    """
    Loads the regression model from disk if available; else returns an error.
    """
    if os.path.exists(model_path):
        print("Loading regression model from disk...")
        with open(model_path, "rb") as f:
            model_lr = pickle.load(f)
        return model_lr
    else:
        raise FileNotFoundError(f"Regression model not found at {model_path}!")

##############################################
# PREDICT MATCH PERCENTAGE FOR EACH JOB
##############################################

def predict_match_for_jobs(resume_info, enriched_df, model_lr, max_experience, embedder):
    """
    For each job in the enriched data (enriched_df), compute features using that job's own
    values (skills, Education, Experience) vs. the resume info.
    Then, predict a match percentage for each job using the regression model.
    Returns a DataFrame with each job title and predicted match percentage.
    """
    results = []
    for idx, row in enriched_df.iterrows():
        # Compute skills similarity: compare resume skills with job's skills
        resume_skills = [s.lower() for s in resume_info.get("skills", [])]
        job_skills = extract_skills(row.get("skills", ""))
        skills_similarity = compute_skills_similarity(job_skills, resume_skills, embedder)
        
        # Compute education match
        resume_edu = resume_info.get("education", "")
        job_edu = row.get("Education", "")
        edu_match = meets_or_exceeds(resume_edu, job_edu)
        
        # Compute experience closeness
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
        
        # Prepare feature vector and predict match score
        features = np.array([[skills_similarity, edu_match, experience_closeness]])
        predicted_match = model_lr.predict(features)[0]
        
        results.append({
            "job_title": row.get("title", ""),
            "predicted_match_percentage": predicted_match,
            "skills_similarity": skills_similarity,
            "education_match": edu_match,
            "experience_closeness": experience_closeness
        })
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="predicted_match_percentage", ascending=False, inplace=True)
    return results_df

##############################################
# MAIN PIPELINE: CONNECT THE MODELS
##############################################

def main():
    # Step 1: Read and extract resume information via OpenAI
    try:
        with open(RESUME_TEXT_PATH, "r", encoding="utf-8") as f:
            resume_text = f.read()
    except Exception as e:
        print("Error reading resume file:", e)
        return
    print("Extracting resume information via OpenAI...")
    resume_info = extract_info_from_resume(resume_text)
    print("Extracted Resume Info:", resume_info)
    validate_resume_info(resume_info)

    # Step 2: Load the Random Forest classifier and predict the cluster
    try:
        classifier = get_classifier_model(MODEL_RF_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    skills_str = ", ".join(resume_info.get("skills", []))
    education = resume_info.get("education", "")
    experience = resume_info.get("years_of_experience", 0)
    sample_df = pd.DataFrame([{
        "most_common_skills": skills_str,
        "most_common_education": education,
        "average_experience": experience
    }])
    predicted_cluster = classifier.predict(sample_df)[0]
    print("Predicted Cluster for the Resume:", predicted_cluster)

    # Step 3: Load clusters summary
    try:
        clusters_df = pd.read_csv(CLUSTERS_SUMMARY_CSV)
    except Exception as e:
        print("Error reading clusters summary CSV:", e)
        return

    cluster_row = clusters_df[clusters_df['cluster'] == predicted_cluster]
    if cluster_row.empty:
        print("No cluster summary found for the predicted cluster.")
        return
    cluster_row = cluster_row.iloc[0].to_dict()
    jobs_list = [j.strip() for j in cluster_row.get("jobs", "").split(";") if j.strip()]
    print("Predicted cluster includes these jobs:", jobs_list)

    # Step 4: Load enriched data
    try:
        enriched_df = pd.read_csv(ENRICHED_DATA_CSV)
    except Exception as e:
        print("Error reading enriched data CSV:", e)
        return

    filtered_enriched_df = enriched_df[enriched_df["title"].str.lower().isin([j.lower() for j in jobs_list])]
    if filtered_enriched_df.empty:
        print("No matching jobs found in the enriched data for the predicted cluster jobs.")
        return
    print(f"Found {len(filtered_enriched_df)} matching job(s) in enriched data.")

    # Step 5: Load the regression model
    try:
        model_lr = get_regression_model(MODEL_LR_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Step 6: Predict matching scores
    results_df = predict_match_for_jobs(resume_info, filtered_enriched_df, model_lr, MAX_EXPERIENCE, embedder)

    # Step 7: Save results
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("\n=== Predicted Match Percentages for Each Job ===")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
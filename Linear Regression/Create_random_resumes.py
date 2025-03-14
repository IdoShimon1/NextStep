import pandas as pd
import numpy as np
import random
import os
from sentence_transformers import SentenceTransformer

# ============================
# USER CONFIGURABLE PARAMETERS
# ============================
# Change these paths to point to your CSV files.
JOB_CSV_PATH = "Data/Enriched_and_cleaned_data.csv"       # Enriched CSV file (with columns: title, skills, Education, Experience)
OUTPUT_CSV_PATH = "Data/Random_Resumes.csv"     # Output file to save synthetic resume-job match data
NUM_SAMPLES = 1000         # Number of synthetic resume-job pairs to generate
MAX_EXPERIENCE = 15        # Maximum years for experience (for normalization)

# ============================
# EDUCATION MAPPING DICTIONARY
# ============================
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

# ============================
# UTILITY FUNCTIONS
# ============================
def extract_skills(skills_str):
    """
    Given a comma-separated string of skills, return a list of cleaned skills.
    If the input is not a string, returns an empty list.
    """
    if not isinstance(skills_str, str):
        return []
    return [skill.strip().lower() for skill in skills_str.split(',') if skill.strip() != ""]

def education_level(edu_str):
    """
    Convert an education string into a numeric level using the education_mapping dictionary.
    Returns the highest level found in the string; if no key is matched, defaults to 0.
    """
    edu_str = edu_str.lower()
    level = -1
    for key, value in education_mapping.items():
        if key in edu_str:
            level = max(level, value)
    return level if level >= 0 else 0

def meets_or_exceeds(resume_edu, job_edu):
    """
    Returns 1 if the resume's education level meets or exceeds the job's requirement, else 0.
    """
    return 1 if education_level(resume_edu) >= education_level(job_edu) else 0

def compute_skills_similarity(job_skills, resume_skills, model):
    """
    Compute the similarity between two lists of skills by:
      1. Computing embeddings for each skill in the lists,
      2. Averaging the embeddings for each list,
      3. Calculating the cosine similarity between the average embeddings,
      4. Mapping the cosine similarity (which can be in [-1,1]) to [0,1].
    """
    if not job_skills or not resume_skills:
        return 0.0
    job_embeddings = model.encode(job_skills, convert_to_numpy=True)
    resume_embeddings = model.encode(resume_skills, convert_to_numpy=True)
    job_mean = np.mean(job_embeddings, axis=0)
    resume_mean = np.mean(resume_embeddings, axis=0)
    dot_product = np.dot(job_mean, resume_mean)
    norm_product = np.linalg.norm(job_mean) * np.linalg.norm(resume_mean)
    similarity = dot_product / norm_product if norm_product != 0 else 0.0
    # Map cosine similarity from [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2
    return similarity

# ============================
# LOAD JOB DATA
# ============================
try:
    jobs_df = pd.read_csv(JOB_CSV_PATH)
except Exception as e:
    print(f"Error reading job CSV file: {e}")
    exit(1)

required_columns = ["title", "skills", "Education", "Experience"]
for col in required_columns:
    if col not in jobs_df.columns:
        raise ValueError(f"Column '{col}' not found in job CSV. Please check your CSV format.")

# ============================
# PREPARE SKILL AND EDUCATION POOLS
# ============================
# Build a pool of all possible skills from job postings.
all_skills = set()
for skills_str in jobs_df['skills']:
    all_skills.update(extract_skills(skills_str))
all_skills = list(all_skills)

# Build a pool of education options from the job postings.
education_pool = list(jobs_df['Education'].unique())

# ============================
# LOAD THE BERT MODEL FOR EMBEDDINGS
# ============================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================
# GENERATE SYNTHETIC RESUME-JOB PAIRS
# ============================
data_rows = []

for i in range(NUM_SAMPLES):
    # Randomly select a job posting.
    job_row = jobs_df.sample(n=1).iloc[0]
    job_title = job_row['title']
    job_skills = extract_skills(job_row['skills'])
    job_education = job_row['Education']
    job_experience = job_row['Experience']
    
    # Generate a synthetic resume.
    num_resume_skills = random.randint(3, min(10, len(all_skills)))
    resume_skills_sample = random.sample(all_skills, num_resume_skills)
    resume_skills_str = ", ".join(resume_skills_sample)
    resume_education = random.choice(education_pool)
    resume_experience = random.randint(0, MAX_EXPERIENCE)
    
    # Compute features.
    skills_similarity = compute_skills_similarity(job_skills, resume_skills_sample, model)
    edu_match = meets_or_exceeds(resume_education, job_education)
    experience_diff = abs(resume_experience - job_experience)
    normalized_experience_diff = experience_diff / MAX_EXPERIENCE  # 0 means perfect, 1 means worst match

    # Compute a match score.
    # We use 50% weight for skills similarity, 30% for education, and 20% for experience closeness.
    raw_score = 0.5 * skills_similarity + 0.3 * edu_match + 0.2 * (1 - normalized_experience_diff)
    match_score = raw_score * 100  # Scale to percentage (0-100)

    data_rows.append({
        "resume_id": i,
        "job_title": job_title,
        "resume_skills": resume_skills_str,
        "resume_education": resume_education,
        "resume_experience": resume_experience,
        "job_skills": ", ".join(job_skills),
        "job_education": job_education,
        "job_experience": job_experience,
        "skills_similarity": skills_similarity,
        "education_match": edu_match,
        "experience_diff": experience_diff,
        "match_score": match_score
    })

synthetic_df = pd.DataFrame(data_rows)

# ============================
# SAVE THE RESULTS
# ============================
try:
    synthetic_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Synthetic resume-job match data saved to: {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Error saving the output CSV: {e}")

print(synthetic_df.head())
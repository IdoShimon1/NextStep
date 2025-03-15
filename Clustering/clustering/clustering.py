import pandas as pd
import numpy as np
import pickle  # For saving the model and embeddings
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
from langdetect import detect
from collections import Counter

# -------------------------
# Load BERT model and tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# -------------------------
# Function to generate BERT embeddings
# -------------------------
def get_bert_embedding(text):
    if not isinstance(text, str):
        text = ""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    outputs = model(**inputs)
    # Take the mean of the last hidden states as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding.flatten()

# -------------------------
# Load and preprocess the data
# (Assuming CSV has columns: title, skills, Education, Experience)
# -------------------------
df = pd.read_csv("data/Enriched_and_cleaned_data.csv")

# Optionally ensure the columns are as expected. If your CSV is guaranteed correct, skip:
df = df[['title', 'skills', 'Education', 'Experience']]

# Combine text fields for embedding
# You can decide which columns to include; here we use title + skills for the BERT text.
df['combined_text'] = (
    df['title'].fillna('') + ' ' +
    df['skills'].fillna('') + ' ' +
    df['Education'].fillna('')
)

# Remove noisy titles if needed (update your list as necessary)
noisy_titles = ['everything', 'professore', 'madrid area spain', 'top_related_titles', 'city head']
df = df[~df['title'].str.contains('|'.join(noisy_titles), na=False)]

# Filter non-English titles (this step can be expensive if your dataset is large)
df = df[df['title'].apply(lambda x: detect(x) == 'en' if isinstance(x, str) else False)]

# -------------------------
# Compute embeddings
# -------------------------
print("Computing BERT embeddings...")
df['embedding'] = df['combined_text'].apply(get_bert_embedding)
X = np.vstack(df['embedding'].values)

# -------------------------
# Clustering
# -------------------------
clustering = AgglomerativeClustering(
    n_clusters=80,     # or None if you use distance_threshold
    distance_threshold=None
).fit(X)

df['cluster'] = clustering.labels_

# Review cluster sizes
cluster_sizes = df['cluster'].value_counts()
print("\nCluster Sizes:")
print(cluster_sizes)

# -------------------------
# Save the updated dataframe with clusters
# -------------------------
df.to_csv("clustered_job_titles.csv", index=False)

# Optionally save the clustering model
with open("clustering_model.pkl", "wb") as f:
    pickle.dump(clustering, f)

# -------------------------
# (Optional) Plot dendrogram
# -------------------------
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(20, 10))
dendrogram(
    linkage_matrix,
    truncate_mode="level",
    p=10,  # Show more levels
    labels=df['title'].values,
    leaf_rotation=90,
    leaf_font_size=10
)
plt.title("Job Titles Clustering Dendrogram")
plt.xlabel("Job Title")
plt.ylabel("Distance")
plt.show()

# -------------------------
# Aggregate each cluster's info:
#   - Cluster Number
#   - List of Jobs
#   - Most Common Skills
#   - Most Common Education
#   - Average Experience
# -------------------------
grouped = df.groupby('cluster')
cluster_rows = []

for cluster_id, group in grouped:
    # 1) The cluster number
    cluster_num = cluster_id
    
    # 2) List of jobs in this cluster
    job_titles = group['title'].unique().tolist()  # or you can keep them as a single string
    job_titles_str = "; ".join(job_titles)  # put them in one string if you prefer
    
    # 3) Most common skills
    # Here we assume 'skills' is a comma-separated string. 
    # We collect all skills from every row in the cluster and find the top frequent ones.
    all_skills = []
    for skill_list in group['skills']:
        if isinstance(skill_list, str):
            # split on commas
            skill_tokens = [s.strip() for s in skill_list.split(',') if s.strip()]
            all_skills.extend(skill_tokens)
    
    # Let's say you want the top 5 most frequent skills
    skills_counter = Counter(all_skills)
    top_skills = skills_counter.most_common(5)
    most_common_skills = [skill for (skill, count) in top_skills]
    most_common_skills_str = ", ".join(most_common_skills)
    
    # 4) Most common education (the single most frequent)
    if not group['Education'].mode().empty:
        most_common_education = group['Education'].mode().iloc[0]
    else:
        most_common_education = None
    
    # 5) Average experience
    # Make sure 'Experience' is numeric
    group['Experience'] = pd.to_numeric(group['Experience'], errors='coerce')
    avg_experience = group['Experience'].mean()
    
    # Create a row dictionary
    row = {
        "cluster": cluster_num,
        "jobs": job_titles_str,
        "most_common_skills": most_common_skills_str,
        "most_common_education": most_common_education,
        "average_experience": round(avg_experience, 2) if pd.notnull(avg_experience) else None
    }
    cluster_rows.append(row)

# -------------------------
# Create summary DataFrame and export to CSV
# -------------------------
summary_df = pd.DataFrame(cluster_rows)
summary_df.to_csv("clusters_summary1.csv", index=False)

# Quick print
print("\nCluster Summary:")
print(summary_df)

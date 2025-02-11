import pandas as pd
import numpy as np
import pickle  # For saving the model and embeddings
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to generate BERT embeddings
def get_bert_embedding(text):
    if not isinstance(text, str):
        text = ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding.flatten()

# Load and preprocess the data
df = pd.read_csv("data/cleaned_job_titles.csv", header=None, names=["title", "pdl_count", "top_related_titles", "skills"])
df['combined_text'] = df['title'] + " " + df['top_related_titles'].fillna('') + " " + df['skills'].fillna('')
df['combined_text'] = df['combined_text'].fillna('').astype(str)

# Generate or load embeddings
embeddings_file = "embeddings.npy"
if os.path.exists(embeddings_file):
    X = np.load(embeddings_file)
else:
    df['embedding'] = df['combined_text'].apply(get_bert_embedding)
    X = np.vstack(df['embedding'].values)
    np.save(embeddings_file, X)  # Save the embeddings

# Perform hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=20).fit(X)
df['cluster'] = clustering.labels_

# Save the updated dataframe with clusters
df.to_csv("clustered_job_titles.csv", index=False)

# Save the clustering model (optional)
with open("clustering_model.pkl", "wb") as f:
    pickle.dump(clustering, f)

# Plot dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, truncate_mode="level", p=5, labels=df['title'].values)
plt.title("Job Titles Clustering Dendrogram")
plt.xlabel("Job Title")
plt.ylabel("Distance")
plt.show()
# Group by cluster and print titles in each cluster
for cluster_id in sorted(df['cluster'].unique()):
    print(f"\nCluster {cluster_id}:")
    cluster_titles = df[df['cluster'] == cluster_id]['title'].values.astype(str)
    print(", ".join(cluster_titles[:10]))  # Print first 10 titles in each cluster
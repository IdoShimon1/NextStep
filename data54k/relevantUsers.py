##script echo skipping
import pandas as pd
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
from langdetect import detect
from collections import Counter
from tqdm import tqdm






##script echo skipping
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
##script echo skipping
number=1

def get_bert_embedding(text):
    global number
    number+=1
    print("currnet" , number)
    if not isinstance(text, str):
        text = ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding.flatten()


##script echo skipping
df_people=pd.read_csv("/Users/nicolegoihman/Desktop/NextStep/01_people.csv")
df_people=df_people[['person_id', 'name']]

##df = df[['title', 'skills', 'Education', 'Experience']]

df_education=pd.read_csv("/Users/nicolegoihman/Desktop/NextStep/03_education.csv")
df_education=df_education[['person_id', 'program']]


df_experience=pd.read_csv("/Users/nicolegoihman/Desktop/NextStep/04_experience.csv")
df_experience=df_experience[['person_id', 'title']]


df_person_skills=pd.read_csv("/Users/nicolegoihman/Desktop/NextStep/05_person_skills.csv")
df_skills=pd.read_csv("/Users/nicolegoihman/Desktop/NextStep/06_skills.csv")

uniqe_lise=[]
for title in df_experience["title"]:
    if title in uniqe_lise:
        continue
    else:
        uniqe_lise.append(title)
print("the uniqe len is:", len(uniqe_lise))

##script echo skipping
print("Computing BERT embeddings...")
tqdm.pandas(desc="Computing BERT embeddings")
df_experience['embedding'] = df_experience['title'].apply(get_bert_embedding)

X = np.vstack(df_experience['embedding'].values)
##script echo skipping

clustering = AgglomerativeClustering(n_clusters=50, distance_threshold=None).fit(X)
df_experience['cluster'] = clustering.labels_

cluster_sizes = df_experience['cluster'].value_counts()

print("\nCluster Sizes:")
print(cluster_sizes)
##script echo skipping
df_experience.to_csv("clustered_job_titles.csv", index=False)

with open("clustering_model.pkl", "wb") as f:
    pickle.dump(clustering, f)
##script echo skipping
linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(20, 10))
dendrogram(linkage_matrix, truncate_mode="level", p=10, labels=df_experience['title'].values, leaf_rotation=90, leaf_font_size=10)
plt.title("Job Titles Clustering Dendrogram")
plt.xlabel("Job Title")
plt.ylabel("Distance")
plt.show()
##script echo skipping
grouped = df_experience.groupby('cluster')
cluster_rows = []

for cluster_id, group in grouped:
    cluster_num = cluster_id
    job_titles = group['title'].unique().tolist()
    job_titles_str = "; ".join(job_titles)
    all_skills = []

    



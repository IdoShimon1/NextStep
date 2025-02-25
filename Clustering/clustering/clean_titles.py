import pandas as pd
import re
from fuzzywuzzy import fuzz, process  # Install with `pip install fuzzywuzzy`
from langdetect import detect  # Install with `pip install langdetect`

# Load the CSV with job titles
df = pd.read_csv("data/job_titles.csv", header=None, names=["title", "pdl_count", "top_related_titles", "skills"])

# Step 1: Clean and Standardize Job Titles
def clean_title(title):
    """Function to clean and standardize job titles."""
    if not isinstance(title, str):
        return ""
    title = title.lower().strip()  # Convert to lowercase and strip whitespace
    title = re.sub(r"[^a-z0-9\s]", "", title)  # Remove special characters
    title = re.sub(r"\b(mgr)\b", "manager", title)  # Replace abbreviations
    title = re.sub(r"\b(analist)\b", "analyst", title)  # Correct common spelling errors
    title = re.sub(r"\s+", " ", title)  # Replace multiple spaces with a single space
    return title

# Step 2: Remove Non-English Titles
def is_english(title):
    """Function to check if a title is in English using langdetect."""
    try:
        return detect(title) == "en"
    except:
        return False

# Apply the cleaning function and remove non-English titles
df['cleaned_title'] = df['title'].apply(clean_title)
df = df[df['cleaned_title'].apply(is_english)]

# Step 3: Deduplication and Fuzzy Matching
unique_titles = {}  # Dictionary to store unique titles
similarity_threshold = 85  # Set similarity threshold for merging titles

for title in df['cleaned_title'].unique():
    if len(unique_titles) == 0:
        unique_titles[title] = [title]
        continue

    # Find the most similar title in unique_titles
    match = process.extractOne(title, unique_titles.keys(), scorer=fuzz.ratio)
    
    if match and match[1] >= similarity_threshold:  # Check if a match is found and meets the threshold
        unique_titles[match[0]].append(title)
    else:
        unique_titles[title] = [title]

# Print the merged groups for review
for key, group in unique_titles.items():
    print(f"{key}: {', '.join(group)}")

# Step 4: Map Cleaned Titles Back to the DataFrame
# Create a mapping from each variation to its standardized form
title_map = {title: key for key, group in unique_titles.items() for title in group}
df['final_title'] = df['cleaned_title'].map(title_map)

# Save the cleaned and updated dataset
df.to_csv("cleaned_job_titles.csv", index=False)

# Step 5: Optional - Show Final Deduplicated Titles
print("\nFinal Deduplicated Titles:")
print(df[['title', 'final_title']].drop_duplicates().head(20))

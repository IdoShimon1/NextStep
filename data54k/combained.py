import pandas as pd

# Load all CSVs
people_df = pd.read_csv("01_people.csv")[["person_id"]]
# ⚠️ Do NOT reduce to a Series — keep it a DataFrame
# people_df = people_df["person_id"]  ❌ Remove this line

education_df = pd.read_csv("03_education.csv")[["person_id", "program"]]
experience_df = pd.read_csv("04_experience.csv")[["person_id", "title"]]
skills_df = pd.read_csv("05_person_skills.csv")[["person_id", "skill"]]

# Group experience and skills by person_id and join as comma-separated strings
experience_grouped = experience_df.groupby("person_id")["title"] \
    .apply(lambda x: ", ".join(x.dropna().astype(str))).reset_index()

skills_grouped = skills_df.groupby("person_id")["skill"] \
    .apply(lambda x: ", ".join(x.dropna().astype(str))).reset_index()

# Get the most recent (last) education record
education_grouped = education_df.groupby("person_id")["program"].last().reset_index()

# Merge everything together
merged = people_df.merge(education_grouped, on="person_id", how="left")
merged = merged.merge(experience_grouped, on="person_id", how="left")
merged = merged.merge(skills_grouped, on="person_id", how="left")

# Rename columns
merged.columns = ["person_id", "education", "experience", "skills"]

# Save to CSV with semicolon separator
merged.to_csv("combined_people.csv", index=False)

import pandas as pd

# Load your dataset
df = pd.read_csv('resume_data.csv')

# First, clean the 'positions' string by removing square brackets
df['positions'] = df['positions'].str.replace(r'[\[\]]', '', regex=True)

df['positions'] = df['positions'].str.lower()


# Then split the cleaned string by comma
df['positions'] = df['positions'].str.split(',')

# Remove extra whitespace from each job title in the list
df['positions'] = df['positions'].apply(lambda jobs: [job.strip() for job in jobs] if isinstance(jobs, list) else jobs)

# Explode the list so that each job becomes a separate row
positions_exploded = df.explode('positions')

# Count the occurrences of each unique job title
position_counts = positions_exploded['positions'].value_counts().reset_index()

# Rename the columns for clarity
position_counts.columns = ['position', 'count']

# Save the counts to a new CSV file
position_counts.to_csv('position_counts.csv', index=False)

print("CSV file 'position_counts.csv' has been created.")

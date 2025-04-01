import pandas as pd

# Load your dataset (update the file name as needed)
df = pd.read_csv('resume_data.csv')

# Clean the job_position_name field:
# Remove square brackets if present and strip any extra whitespace.
df['job_position_name'] = df['job_position_name'].str.replace(r'[\[\]]', '', regex=True).str.strip()
df['job_position_name'] = df['job_position_name'].str.lower()


# Count the occurrences of each unique job_position_name
job_counts = df['job_position_name'].value_counts().reset_index()

# Rename the columns for clarity
job_counts.columns = ['job_position_name', 'count']

# Save the counts to a new CSV file
job_counts.to_csv('job_position_counts.csv', index=False)

print("CSV file 'job_position_counts.csv' has been created.")

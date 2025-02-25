import pandas as pd
import requests
import json
import time
import os

# Your API key and OpenAI API endpoint
api_key = ""
url = "https://api.openai.com/v1/chat/completions"

# Load and clean the CSV
file_path = "Data\\job_titles.csv"
df = pd.read_csv(file_path, on_bad_lines='skip')
df['skills'] = df.iloc[:, 2:].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)
df = df[['title', 'skills']]

# Output file to track processed results
output_file = "Enriched_and_cleaned_data.csv"
processed_titles = set()

if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file)
    processed_titles = set(processed_df['title'].unique())


def generate_education_and_experience(title, skills, retries=5):
    prompt = f"""
    Job Title: {title}
    Skills: {skills}

    Task 1: Is the job title in English? Answer "Yes" or "No".
    Task 2: If the answer to Task 1 is "Yes", check if:
        - The job title is a legitimate, professional role that someone could apply for on LinkedIn 
        (e.g., "Software Engineer," "Teacher," "Project Manager," etc.).
        - The job title is NOT a personal status or non-professional role (like 'mom', 'dad', 'student', 'retired', 'housewife', 'mother', etc.).
        - The listed "Skills" logically match that legitimate job title.
    If the job title fails any of these checks, return "In English: No".

    Task 3: If the answer to Task 2 is still "Yes", suggest the appropriate education requirement for this job, exactly one degree (e.g., "Bachelor's Degree in Computer Science"). Do not say phrases like "in relevant field."

    Task 4: If the answer to Task 2 is still "Yes", suggest the appropriate years of experience required, only as an integer.

    Finally, return your answers in exactly this format:
    "In English: [Yes/No]; Education: [your answer]; Experience: [your answer]".

    If the job is not in English, not a valid LinkedIn role, or the skills are irrelevant to that role, return:
    "In English: No".
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "max_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content']
            
            print(f"AI Response for {title}: {response_text}")

            if "In English: No" in response_text:
                print(f"Skipping non-English job title: {title}")
                return None

            education, experience = "None", "None"
            if "Education:" in response_text and "Experience:" in response_text:
                try:
                    education = response_text.split("Education:")[1].split(";")[0].strip()
                    experience = response_text.split("Experience:")[1].strip()
                except IndexError:
                    pass

            return education, experience

        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded for '{title}'. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Request error for title '{title}': {e}")
                return None

    print(f"Failed to process title '{title}' after {retries} attempts.")
    return None


# Main function to process all rows one by one
def main():
    total_rows = len(df)
    for index, row in df.iterrows():
        title = row['title']
        
        if title in processed_titles:
            print(f"Skipping already processed title: {title}")
            continue

        skills = row['skills']
        result = generate_education_and_experience(title, skills)

        if result:
            education, experience = result
            row['Education'] = education
            row['Experience'] = experience

            # Save to the CSV file
            if not os.path.exists(output_file):
                pd.DataFrame([row]).to_csv(output_file, index=False)
            else:
                pd.DataFrame([row]).to_csv(output_file, mode='a', header=False, index=False)
        
        print(f"Processed row {index + 1} of {total_rows}")


# Run the main function
if __name__ == "__main__":
    main()

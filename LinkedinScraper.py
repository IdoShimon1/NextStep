import tkinter as tk
from tkinter import messagebox
import requests
import json
import os
import time

# --------------------------
# Configuration
# --------------------------
API_KEY = os.getenv("OPENAI_API_KEY")    
URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
OUTPUT_FILE = "linkedin_profiles.json"

# --------------------------
# OpenAI API Call Function
# --------------------------
def process_linkedin_data(uid, raw_data, retries=5):
    prompt = f"""
I am scraping LinkedIn profiles of tech professionals. I need you to extract and clean the data from the raw input according to these instructions:

1. **Education:** Extract all education entries. For each entry, output:
   - degree
   - field
   - institution
   - year_completed (if available)
   If a high school is mentioned, simply output "High school" (without the specific name).

2. **Job History:** Extract all job history entries from the "Experience" section. For each entry, output:
   - job_id (assign sequential IDs starting from "001" in the order they appear, with the most recent job first)
   - title
   - company
   - start_date (format YYYY-MM-DD if possible)
   - end_date (format YYYY-MM-DD if available; otherwise, use null)
   - skills: list any tech-related skills mentioned in that job description.
   Remove any non-tech related roles or skills.

3. **Skills:** Extract all unique tech-related skills mentioned anywhere in the profile and list them in the "skills" array.

4. **Label:** In the "label" field, put the user's most recent (latest) job title if it belongs to the tech category. If not, leave it empty.

5. **Overall:** Remove any data not related to tech for both job history and skills.

Return ONLY valid JSON. **Do not** include code fences, markdown, or any additional text.

Your JSON must follow exactly this format:
{{
  "uid": "{uid}",
  "label": "user's most recent tech job title",
  "education": [
    {{
      "degree": "Degree",
      "field": "Field",
      "institution": "Institution",
      "year_completed": "Year"
    }}
    // ... more education entries
  ],
  "job_history": [
    {{
      "job_id": "001",
      "title": "Job Title",
      "company": "Company Name",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD" or null,
      "skills": ["skill1", "skill2", ...]
    }}
    // ... more job entries
  ],
  "skills": ["unique", "tech", "skills", ...]
}}

User Data:
{raw_data}

Ensure that the output is valid JSON with the uid exactly set to "{uid}" and no extra text or formatting.
"""

    data = {
        "model": "gpt-4o-mini",  # Change this if desired (e.g., "gpt-4")
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    for attempt in range(retries):
        try:
            response = requests.post(URL, headers=HEADERS, data=json.dumps(data))
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            return content
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                wait_time = 2 ** attempt  # exponential backoff
                messagebox.showinfo("Rate limit", f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                messagebox.showerror("Request Error", f"Error: {e}")
                return None
    messagebox.showerror("Failure", "Failed to process the data after multiple attempts.")
    return None

# --------------------------
# Post-Processing: Update Label
# --------------------------
def update_label_with_latest_job(output_json):
    """Updates the 'label' field with the title of the latest job.
       If an ongoing job (end_date is null) exists, use its title.
       Otherwise, choose the job with the most recent start_date.
    """
    job_history = output_json.get("job_history", [])
    if job_history:
        # Filter for jobs that are ongoing (end_date is null)
        current_jobs = [job for job in job_history if job.get("end_date") is None]
        if current_jobs:
            # If multiple, choose the one with the latest start_date
            latest_job = max(current_jobs, key=lambda job: job.get("start_date", ""))
            output_json["label"] = latest_job.get("title", "")
        else:
            # Otherwise, choose the job with the latest start_date
            latest_job = max(job_history, key=lambda job: job.get("start_date", ""))
            output_json["label"] = latest_job.get("title", "")
    return output_json

# --------------------------
# File Handling Functions
# --------------------------
def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading existing JSON: {e}")
            return []
    return []

def save_data(data_list):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data_list, f, indent=4)

def get_next_uid():
    data_list = load_existing_data()
    uid = 500
    if data_list:
        try:
            existing_uids = [int(item.get("uid", 500)) for item in data_list]
            uid = max(existing_uids) + 1
        except Exception:
            uid = 500
    return uid

# --------------------------
# GUI Functions
# --------------------------
def submit_data(event=None):
    raw_data = text_input.get("1.0", tk.END).strip()
    if not raw_data:
        messagebox.showinfo("No Data", "Please paste some LinkedIn data into the text box.")
        return

    uid = get_next_uid()
    messagebox.showinfo("Processing", f"Processing data with UID {uid}...")
    output_text = process_linkedin_data(uid, raw_data)

    if not output_text:
        return

    # --- NEW: Remove any accidental code fences if they appear. ---
    json_text = output_text.strip()
    if json_text.startswith("```"):
        # Split into lines
        lines = json_text.split("\n")
        # Remove the first and last line if they contain triple backticks
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        json_text = "\n".join(lines).strip()
    # --- END of code-fence removal ---

    # Attempt to parse the final string as JSON
    try:
        output_json = json.loads(json_text)
    except json.JSONDecodeError:
        messagebox.showerror("JSON Error",
                             "Failed to parse JSON from the API response. "
                             f"Raw output:\n{output_text}")
        return

    # Ensure there is a "label" key in the JSON output.
    if "label" not in output_json:
        output_json["label"] = ""
    
    # Update the label with the latest job title from job_history.
    output_json = update_label_with_latest_job(output_json)

    data_list = load_existing_data()
    data_list.append(output_json)
    save_data(data_list)
    messagebox.showinfo("Success", f"Data processed and saved with UID {uid}.")
    text_input.delete("1.0", tk.END)

# --------------------------
# Main GUI Setup
# --------------------------
root = tk.Tk()
root.title("LinkedIn Profile Scraper")

# Create a frame for the text widget and scrollbar
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a vertical scrollbar
scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a text box with the scrollbar attached
text_input = tk.Text(frame, height=20, width=80, yscrollcommand=scrollbar.set)
text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=text_input.yview)

# Bind the Enter key (Return) to submission.
# Pressing Enter will trigger submission instead of inserting a newline.
text_input.bind("<Return>", lambda event: (submit_data(), "break"))

# Create a Submit button (in case you prefer clicking)
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.pack(pady=(0, 10))

root.mainloop()

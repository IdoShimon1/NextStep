import pandas as pd
import csv

# טעינת הנתונים מקובץ CSV
df = pd.read_csv('nn/resume_data.csv')

# ניקוי עמודת 'positions':
# הסרת סוגריים מרובעים, המרה לאותיות קטנות ופיצול לפי פסיק.
df['positions'] = df['positions'].str.replace(r'[\[\]]', '', regex=True)
df['positions'] = df['positions'].str.lower().str.split(',')

# ניקוי עמודת 'job_position_name':
# הסרת סוגריים מרובעים, הסרת רווחים מיותרים והמרה לאותיות קטנות.
df['job_position_name'] = (
    df['job_position_name']
    .str.replace(r'[\[\]]', '', regex=True)
    .str.strip()
    .str.lower()
)

# אתחול רשימה שתאחסן את התוצאות
final_data = []

# קבוצת הנתונים לפי job_position_name
for job, group in df.groupby('job_position_name'):
    unique_positions = set()
    # עבור כל שורה בקבוצה, נבדוק את העמודה 'positions'
    for positions_list in group['positions']:
        if isinstance(positions_list, list):
            # הסרת רווחים מיותרים בכל ערך והרחקת ערכים ריקים
            cleaned_positions = [p.strip() for p in positions_list if p.strip() != ""]
            unique_positions.update(cleaned_positions)
    # הוספת השורה לרשימת התוצאות: שם המשרה ומספר הערכים הייחודיים
    final_data.append([job, len(unique_positions)])

# כתיבת התוצאות לקובץ CSV
with open('unique_per_job.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["job_position_name", "count"])  # כותרות העמודות
    writer.writerows(final_data)

import json
from pathlib import Path

# --------- helper to map UID ➔ label ---------
def map_label(uid: int) -> str:
    ranges = [
        (500, 549, "Data Analyst"),
        (550, 599, "Data Scientist"),
        (600, 649, "Data Engineer"),
        (650, 699, "Machine Learning Engineer"),
        (700, 749, "Chief Data Officer"),
        (750, 799, "QA Engineer"),
        (800, 824, "Fullstack"),
        (825, 849, "Frontend"),
        (850, 874, "Backend"),
        (875, 907, "PT"),
        (908, 941, "CISO"),
    ]
    for low, high, label in ranges:
        if low <= uid <= high:
            return label
    return "Unknown"            # fallback (UID outside all ranges)

# --------- main cleaning routine ---------
def clean_records(
    in_file: str | Path = "linkedin_profiles.json",
    out_file: str | Path = "cleaned_data.json",
) -> None:
    with open(in_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        # 1️⃣  Drop first job if its end_date is null/None/empty
        jobs = rec.get("job_history", [])
        if jobs and jobs[0].get("end_date") in (None, "", "null"):
            jobs.pop(0)

        # 2️⃣  Update label based on UID
        rec["label"] = map_label(int(rec["uid"]))

    # 3️⃣  Write the cleaned data
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"✅ Cleaned file saved to {out_file}")

# --------- run from CLI ---------
if __name__ == "__main__":
    clean_records()

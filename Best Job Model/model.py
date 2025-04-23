# ------------------------------------------------------------
# 0.  (Optional) install/upgrade once:
# !pip install -U "transformers[torch]" datasets torch scikit-learn pandas matplotlib --quiet
# ------------------------------------------------------------
"""
Script: resume_title_classifier_without_excluded_labels.py
Adds rich evaluation metrics and visualisations (confusion matrix, precision/recall/F1) to a
BERT résumé‑title classifier **while excluding specific labels**.

Standalone usage is identical to the original script.  After training it will:
  • Evaluate on the test set and print accuracy + full classification report
  • Plot & save a confusion‑matrix graph (PNG) into the output directory

The following job‑title labels are **ignored** during training/evaluation:
  • Security Engineer
  • Application Security Engineer
  • Frontend software engineer
"""

import json, inspect, pickle, os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset, DatasetDict

# ------------------------------------------------------------
# 1.  Load & flatten résumé JSON ------------------------------
DATA_PATH = "ModelDev/Files/data_merged.json"     # <‑‑ YOUR JSON
EXCLUDE_LABELS = {
    "Security Engineer",
    "Application Security Engineer",
    "Frontend software engineer",
}

with open(DATA_PATH, encoding="utf-8") as fp:
    raw_data = json.load(fp)

# ----- Filter out unwanted labels -----
initial_count = len(raw_data)
raw_data = [sample for sample in raw_data if sample.get("label") not in EXCLUDE_LABELS]
print(
    f"Filtered out {initial_count - len(raw_data)} samples spanning {len(EXCLUDE_LABELS)} labels: {', '.join(EXCLUDE_LABELS)}"
)
print(f"Remaining samples: {len(raw_data)}")


def join_list(x):
    return " ".join(x) if isinstance(x, list) else str(x)


def resume_to_text(sample: dict) -> str:
    """Concatenate education + job history (skipping the first job **if** it's ongoing) + skills."""
    parts = []

    # ---------- Education ----------
    for ed in sample.get("education", []):
        parts.append(
            f"{ed.get('degree','')} {ed.get('field','')} {ed.get('institution','')}"
        )

    # ---------- Job history ----------
    jobs = sample.get("job_history", [])

    # If the first job has end_date == None, drop it (it's the current job)
    if jobs and jobs[0].get("end_date") is None:
        jobs = jobs[1:]

    for job in jobs:
        parts.append(f"{job.get('title','')} {job.get('company','')}")

    # ---------- Skills ----------
    parts.append(join_list(sample.get("skills", [])))

    return " ".join(parts)


texts  = [resume_to_text(s) for s in raw_data]
labels = [s["label"] for s in raw_data]

# ------------------------------------------------------------
# 2.  Label‑encode job titles --------------------------------
le = LabelEncoder()
y  = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Detected {num_classes} unique job titles after exclusion.")

# ------------------------------------------------------------
# 3.  70 / 15 / 15 split with rarity guard --------------------
min_count = min(Counter(y).values())
print("Smallest class size:", min_count)
strat = y if min_count >= 3 else None
if strat is None:
    print("⚠️  Some classes have <3 samples – using random (non‑stratified) split.")

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, y, test_size=0.30, stratify=strat, random_state=42
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=y_temp if strat is not None else None,
    random_state=42
)
print(f"Train/valid/test sizes: {len(X_train)}/{len(X_valid)}/{len(X_test)}")

# ------------------------------------------------------------
# 4.  Build 🤗 Dataset ----------------------------------------
ds = DatasetDict({
    "train": Dataset.from_dict({"text": X_train, "label": y_train}),
    "valid": Dataset.from_dict({"text": X_valid, "label": y_valid}),
    "test" : Dataset.from_dict({"text": X_test , "label": y_test }),
})

# ------------------------------------------------------------
# 5.  Tokeniser & encoding -----------------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 256

print("\nLoading tokenizer…")
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer for {MODEL_NAME}: {e}")


def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

print("Tokenising…")
ds_tok   = ds.map(tokenize, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tok, return_tensors="pt")

# ------------------------------------------------------------
# 6.  Model ---------------------------------------------------
print("\nLoading model…")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes
)

# ------------------------------------------------------------
# 7.  TrainingArguments — fully adaptive ---------------------
set_seed(42)
TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)

args_kwargs = dict(
    output_dir="bert_resume_title",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    # ---------- NEW: make Windows happy ----------
    save_safetensors=False,          # <‑‑ disables .safetensors → avoids mmap lock
    overwrite_output_dir=True,       # <‑‑ convenience: re‑run without deleting dir
)

# These adaptive options keep your original behaviour
if {"evaluation_strategy", "save_strategy"}.issubset(TA_PARAMS):
    args_kwargs.update(evaluation_strategy="epoch", save_strategy="epoch")
    if "load_best_model_at_end" in TA_PARAMS:
        args_kwargs.update(load_best_model_at_end=True)
    if "metric_for_best_model" in TA_PARAMS:
        args_kwargs.update(metric_for_best_model="accuracy")
elif "evaluate_during_training" in TA_PARAMS:
    args_kwargs.update(evaluate_during_training=True)

train_args = TrainingArguments(**args_kwargs)

# ------------------------------------------------------------
# 8.  Trainer & training -------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

trainer = Trainer(
    model           = model,
    args            = train_args,
    train_dataset   = ds_tok["train"],
    eval_dataset    = ds_tok["valid"],
    tokenizer       = tok,
    data_collator   = collator,
    compute_metrics = compute_metrics,
)

print("\nTraining…")
trainer.train()

# ------------------------------------------------------------
# 9.  Evaluate — accuracy + full report ----------------------
print("\nEvaluating on test set…")
metrics = trainer.evaluate(ds_tok["test"])
acc = metrics.get("eval_accuracy", metrics.get("accuracy", 0))
print(f"Test accuracy: {acc:.3f}\n")

# Detailed classification report
pred_results = trainer.predict(ds_tok["test"])
preds = pred_results.predictions.argmax(axis=1)
print(classification_report(y_test, preds, target_names=le.classes_))

# ------------------------------------------------------------
# 10.  Confusion‑matrix plot ---------------------------------
print("Plotting confusion matrix…")
cm = confusion_matrix(y_test, preds, labels=range(num_classes))
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    include_values=True, cmap="Blues", ax=ax, xticks_rotation=45
)
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()

# Ensure output dir exists & save
os.makedirs(train_args.output_dir, exist_ok=True)
conf_path = os.path.join(train_args.output_dir, "confusion_matrix.png")
plt.savefig(conf_path, dpi=200)
print(f"✔️  Confusion‑matrix image saved → {conf_path}\n")
plt.show()

# ------------------------------------------------------------
# 11.  Save model artefacts ----------------------------------
print("Saving model & artefacts…")
trainer.save_model("bert_resume_title")              # Windows‑safe now
tok.save_pretrained("bert_resume_title/tokenizer")
with open("label_encoder.pkl", "wb") as fp:
    pickle.dump(le, fp)
print("All done! ✔️\n")

# ------------------------------------------------------------
# 12.  Inference helper --------------------------------------

def predict_title(resume_json: dict):
    """Return (predicted_title, probability_vector) **excluding disallowed labels**."""
    model.eval()
    txt = resume_to_text(resume_json)
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]
    return le.inverse_transform([probs.argmax()])[0], probs

if __name__ == "__main__":
    title, _ = predict_title(raw_data[0])
    print(f"Example prediction (uid {raw_data[0]['uid']}): {title}")

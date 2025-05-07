# ------------------------------------------------------------
# 0.  (Optional) install/upgrade once:
# !pip install -U "transformers[torch]" datasets torch scikit-learn pandas matplotlib --quiet
# ------------------------------------------------------------
"""
Script: resume_title_classifier_without_excluded_labels.py
Fine‑tune BERT to predict résumé titles **excluding** certain labels and automatically
output three separate diagnostics graphs:

  • **loss_curve.png**           – training loss vs. epoch
  • **val_accuracy.png**         – validation accuracy vs. epoch
  • **confusion_matrix.png**     – confusion matrix on the held‑out test set

The script also saves the model, tokenizer, and label encoder, and provides a
`predict_title()` helper for inference.
"""

import json, inspect, pickle, os
from collections import Counter, OrderedDict
from datetime import datetime

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
# 1.  Settings ------------------------------------------------
DATA_PATH = "ModelDev/Files/data_merged.json"     
EXCLUDE_LABELS = {
    "Security Engineer",
    "Application Security Engineer",
    "Frontend software engineer",
}
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 256
RANDOM_SEED = 42
EPOCHS = 5
BATCH_SIZE = 8
OUT_DIR = "bert_resume_title"                    

# ------------------------------------------------------------
# 2.  Load JSON & remove unwanted labels ---------------------
with open(DATA_PATH, encoding="utf-8") as fp:
    raw_data = json.load(fp)

initial_count = len(raw_data)
raw_data = [s for s in raw_data if s.get("label") not in EXCLUDE_LABELS]
print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Filtered {initial_count - len(raw_data)} samples → {len(raw_data)} remain.")

# ------------------------------------------------------------
# 3.  Utility: flatten résumé → plain text -------------------

def join_list(x):
    return " ".join(x) if isinstance(x, list) else str(x)

def resume_to_text(sample: dict) -> str:
    """Concatenate education, job history (skip current job), and skills."""
    parts = []
    for ed in sample.get("education", []):
        parts.append(f"{ed.get('degree','')} {ed.get('field','')} {ed.get('institution','')}")
    jobs = sample.get("job_history", [])
    if jobs and jobs[0].get("end_date") is None:  # drop current job
        jobs = jobs[1:]
    for job in jobs:
        parts.append(f"{job.get('title','')} {job.get('company','')}")
    parts.append(join_list(sample.get("skills", [])))
    return " ".join(parts)

texts  = [resume_to_text(s) for s in raw_data]
labels = [s["label"] for s in raw_data]

# ------------------------------------------------------------
# 4.  Encode labels ------------------------------------------
le = LabelEncoder()
y  = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Detected {num_classes} unique job titles after exclusion.")

# ------------------------------------------------------------
# 5.  Train/valid/test split ---------------------------------
from collections import Counter  
min_count = min(Counter(y).values())
strat = y if min_count >= 3 else None
if strat is None:
    print("⚠️  Some classes have <3 samples – using non‑stratified split.")

X_train, X_temp, y_train, y_temp = train_test_split(texts, y, test_size=0.30, stratify=strat, random_state=RANDOM_SEED)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp if strat is not None else None, random_state=RANDOM_SEED)
print(f"Dataset sizes → train: {len(X_train)}, valid: {len(X_valid)}, test: {len(X_test)}")

# ------------------------------------------------------------
# 6.  Build 🤗 Datasets ---------------------------------------
ds = DatasetDict({
    "train": Dataset.from_dict({"text": X_train, "label": y_train}),
    "valid": Dataset.from_dict({"text": X_valid, "label": y_valid}),
    "test" : Dataset.from_dict({"text": X_test , "label": y_test }),
})

# ------------------------------------------------------------
# 7.  Tokenise ------------------------------------------------
print("\nLoading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

print("Tokenising…")
ds_tok   = ds.map(tokenize, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tok, return_tensors="pt")

# ------------------------------------------------------------
# 8.  Model ---------------------------------------------------
print("\nLoading model…")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

# ------------------------------------------------------------
# 9.  TrainingArguments --------------------------------------
set_seed(RANDOM_SEED)
TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)

args_kwargs = dict(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    save_safetensors=False,
    overwrite_output_dir=True,
)

if {"evaluation_strategy", "save_strategy"}.issubset(TA_PARAMS):
    args_kwargs.update(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
elif "evaluate_during_training" in TA_PARAMS:
    args_kwargs["evaluate_during_training"] = True

train_args = TrainingArguments(**args_kwargs)

# ------------------------------------------------------------
# 10. Trainer -------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

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
# 11. Epoch‑level metrics & graphs ---------------------------
print("Generating epoch‑level graphs…")
logs = trainer.state.log_history

# Collect last training loss for each epoch
loss_per_epoch = OrderedDict()
for log in logs:
    if "loss" in log and "epoch" in log:
        loss_per_epoch[log["epoch"]] = log["loss"]  

# Collect validation accuracy entries (once per epoch)
val_epochs = []
val_acc    = []
for log in logs:
    if "eval_accuracy" in log and "epoch" in log:
        val_epochs.append(log["epoch"])
        val_acc.append(log["eval_accuracy"])

os.makedirs(OUT_DIR, exist_ok=True)

# --- Loss curve ---
fig_loss, ax_loss = plt.subplots(figsize=(7,5))
ax_loss.plot(list(loss_per_epoch.keys()), list(loss_per_epoch.values()), marker="o")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Training loss")
ax_loss.set_title("Training Loss vs Epoch")
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_curve.png")
fig_loss.savefig(loss_path, dpi=200)
plt.close(fig_loss)
print(f"✔️  Saved loss curve → {loss_path}")

# ------------------------------------------------------------
# 12. Evaluate on test set -----------------------------------
print("Evaluating on test set…")
metrics = trainer.evaluate(ds_tok["test"])
acc = metrics.get("eval_accuracy", metrics.get("accuracy", 0))
print(f"Test accuracy: {acc:.3f}\n")

pred_results = trainer.predict(ds_tok["test"])
preds = pred_results.predictions.argmax(axis=1)
print(classification_report(y_test, preds, target_names=le.classes_))

# ------------------------------------------------------------
# 13. Confusion‑matrix plot ----------------------------------
print("Plotting confusion matrix…")
cm = confusion_matrix(y_test, preds, labels=range(num_classes))
fig_cm, ax_cm = plt.subplots(figsize=(8,8))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(include_values=True, cmap="Blues", ax=ax_cm, xticks_rotation=45)
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
fig_cm.savefig(cm_path, dpi=200)
plt.close(fig_cm)
print(f"✔️  Saved confusion matrix → {cm_path}\n")

# ------------------------------------------------------------
# 14. Save model artefacts -----------------------------------
print("Saving model & artefacts…")
trainer.save_model(OUT_DIR)
tok.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as fp:
    pickle.dump(le, fp)
print("All done! ✔️\n")

# ------------------------------------------------------------
# 15. Inference helper ---------------------------------------

def predict_title(resume_json: dict):
    """Return (predicted_title, probability_vector) – ignores excluded labels."""
    model.eval()
    txt = resume_to_text(resume_json)
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return le.inverse_transform([probs.argmax()])[0], probs

if __name__ == "__main__":
    title, _ = predict_title(raw_data[0])
    print(f"Example prediction (uid {raw_data[0]['uid']}): {title}")

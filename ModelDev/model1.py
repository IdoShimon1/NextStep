# ------------------------------------------------------------
# resume_title_experiments.py
# ------------------------------------------------------------
"""
Fine–tune **any 🤗 transformer classifier** on résumé–title data and run a
*set of experiments* (different models / hyper‑parameters) with **one command**.

Key features
------------
* 🧩  **Plug‑and‑play models** – supply model names in a list (e.g. BERT, DistilBERT, RoBERTa).
* 🔄  **Grid search** – specify learning‑rates / batch‑sizes and the script
   tries every combination.
* 📝  **Human‑readable output** – each run saves:
    * training‑loss ⇢ *loss_curve.png*
    * validation‑accuracy ⇢ *val_accuracy.png*
    * confusion‑matrix ⇢ *confusion_matrix.png*
    * a JSON summary ⇢ *metrics.json*
* 🏃  **CLI** – override anything from the command‑line, no code‑editing needed.

Minimal example
---------------
```bash
python resume_title_experiments.py \
  --data ModelDev/Files/data_merged.json \
  --exclude "Security Engineer" "Frontend software engineer" \
  --models bert-base-uncased distilbert-base-uncased roberta-base \
  --lr 2e-5 5e-5 \
  --bs 8 16 \
  --epochs 4 6
```
The above tests **3 models × 2 learning‑rates × 2 batch‑sizes × 2 epoch counts = 24 runs**
(likely in ~2‑3 GPU hours rather than days).

Tip: use the `--dry-run` flag first to print every planned run without training.
"""

from __future__ import annotations
import argparse, json, os, pickle, inspect, itertools, shutil
from collections import Counter, OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0.  Configuration dataclass --------------------------------

@dataclass
class ExperimentConfig:
    """Hyper‑parameters for a single training run."""

    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 5
    max_len: int = 256
    weight_decay: float = 0.01
    seed: int = 42
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    metric_for_best_model: str = "accuracy"
    load_best_model_at_end: bool = True

    def tag(self) -> str:
        """Create a short folder‑name like 'bert-base-uncased_lr2e-5_bs8_ep5'."""
        bits = [
            self.model_name.replace("/", "-"),
            f"lr{self.learning_rate:g}",
            f"bs{self.batch_size}",
            f"ep{self.epochs}",
        ]
        return "_".join(bits)

# ------------------------------------------------------------
# 1.  Utility functions --------------------------------------

join_list = lambda x: " ".join(x) if isinstance(x, list) else str(x)

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
    parts.append(" ".join(sample.get("skills", [])))
    return " ".join(parts)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ------------------------------------------------------------
# 2.  Training helper ----------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    texts: List[str],
    labels: List[int],
    le: LabelEncoder,
    out_root: str,
):
    set_seed(cfg.seed)

    # Split data --------------------------------------------------
    min_count = min(Counter(labels).values())
    strat = labels if min_count >= 3 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, stratify=strat, random_state=cfg.seed
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp if strat is not None else None,
        random_state=cfg.seed,
    )

    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"text": X_train, "label": y_train}),
            "valid": Dataset.from_dict({"text": X_valid, "label": y_valid}),
            "test": Dataset.from_dict({"text": X_test, "label": y_test}),
        }
    )

    # Tokeniser & model -----------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    ds_tok = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=cfg.max_len),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorWithPadding(tok, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(le.classes_)
    )

    # TrainingArgs ----------------------------------------------
    TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)

    args_kwargs = dict(
        output_dir=os.path.join(out_root, cfg.tag()),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio if "warmup_ratio" in TA_PARAMS else 0.0,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=50,
        save_safetensors=False if "save_safetensors" in TA_PARAMS else None,
        overwrite_output_dir=True,
    )

    # Add newer arguments only if supported
    if {"evaluation_strategy", "save_strategy"}.issubset(TA_PARAMS):
        args_kwargs.update(
            evaluation_strategy=cfg.evaluation_strategy,
            save_strategy=cfg.save_strategy,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
        )
    elif "evaluate_during_training" in TA_PARAMS:
        args_kwargs["evaluate_during_training"] = True

    if "lr_scheduler_type" in TA_PARAMS:
        args_kwargs["lr_scheduler_type"] = cfg.lr_scheduler_type

    if "fp16" in TA_PARAMS:
        args_kwargs["fp16"] = cfg.fp16

    train_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["valid"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Epoch‑level diagnostics ------------------------------------
    logs = trainer.state.log_history
    _plot_curves(logs, trainer.args.output_dir)

    # Final evaluation -------------------------------------------
    metrics = trainer.evaluate(ds_tok["test"])
    preds = trainer.predict(ds_tok["test"]).predictions.argmax(axis=1)

    _plot_confusion_matrix(
        y_test,
        preds,
        le.classes_,
        os.path.join(trainer.args.output_dir, "confusion_matrix.png"),
    )

    # Save artefacts ---------------------------------------------
    tok.save_pretrained(os.path.join(trainer.args.output_dir, "tokenizer"))
    with open(os.path.join(trainer.args.output_dir, "label_encoder.pkl"), "wb") as fp:
        pickle.dump(le, fp)

    # Store metrics JSON -----------------------------------------
    with open(os.path.join(trainer.args.output_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"✅  Finished {cfg.tag()} – acc={metrics.get('eval_accuracy', 0):.3f}\n")
    return metrics

def run_experiment_with_details(
    cfg: ExperimentConfig,
    texts: List[str],
    labels: List[int],
    le: LabelEncoder,
    out_root: str,
):
    """Run a single experiment and return trainer, metrics, and predictions for detailed analysis."""
    set_seed(cfg.seed)

    # Split data --------------------------------------------------
    min_count = min(Counter(labels).values())
    strat = labels if min_count >= 3 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, stratify=strat, random_state=cfg.seed
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp if strat is not None else None,
        random_state=cfg.seed,
    )

    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"text": X_train, "label": y_train}),
            "valid": Dataset.from_dict({"text": X_valid, "label": y_valid}),
            "test": Dataset.from_dict({"text": X_test, "label": y_test}),
        }
    )

    # Tokeniser & model -----------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    ds_tok = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=cfg.max_len),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorWithPadding(tok, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(le.classes_)
    )

    # TrainingArgs ----------------------------------------------
    TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)

    args_kwargs = dict(
        output_dir=os.path.join(out_root, cfg.tag()),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio if "warmup_ratio" in TA_PARAMS else 0.0,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=50,
        save_safetensors=False if "save_safetensors" in TA_PARAMS else None,
        overwrite_output_dir=True,
    )

    # Add newer arguments only if supported
    if {"evaluation_strategy", "save_strategy"}.issubset(TA_PARAMS):
        args_kwargs.update(
            evaluation_strategy=cfg.evaluation_strategy,
            save_strategy=cfg.save_strategy,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
        )
    elif "evaluate_during_training" in TA_PARAMS:
        args_kwargs["evaluate_during_training"] = True

    if "lr_scheduler_type" in TA_PARAMS:
        args_kwargs["lr_scheduler_type"] = cfg.lr_scheduler_type

    if "fp16" in TA_PARAMS:
        args_kwargs["fp16"] = cfg.fp16

    train_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["valid"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Epoch‑level diagnostics ------------------------------------
    logs = trainer.state.log_history
    _plot_curves(logs, trainer.args.output_dir)

    # Final evaluation -------------------------------------------
    metrics = trainer.evaluate(ds_tok["test"])
    predictions = trainer.predict(ds_tok["test"])

    _plot_confusion_matrix(
        y_test,
        predictions.predictions.argmax(axis=1),
        le.classes_,
        os.path.join(trainer.args.output_dir, "confusion_matrix.png"),
    )

    # Save artefacts ---------------------------------------------
    tok.save_pretrained(os.path.join(trainer.args.output_dir, "tokenizer"))
    with open(os.path.join(trainer.args.output_dir, "label_encoder.pkl"), "wb") as fp:
        pickle.dump(le, fp)

    # Store metrics JSON -----------------------------------------
    with open(os.path.join(trainer.args.output_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"✅  Finished {cfg.tag()} – acc={metrics.get('eval_accuracy', 0):.3f}\n")
    return trainer, metrics, predictions

# ------------------------------------------------------------
# 3.  Plot helpers -------------------------------------------

def _plot_curves(logs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    loss, epochs, v_epochs, v_acc = OrderedDict(), [], [], []
    for log in logs:
        if "loss" in log and "epoch" in log:
            loss[log["epoch"]] = log["loss"]
        if "eval_accuracy" in log and "epoch" in log:
            v_epochs.append(log["epoch"])
            v_acc.append(log["eval_accuracy"])

    # Loss
    _plot_single_curve(
        list(loss.keys()),
        list(loss.values()),
        "Epoch",
        "Training loss",
        "Training Loss vs Epoch",
        os.path.join(out_dir, "loss_curve.png"),
    )

    # Val‑accuracy
    _plot_single_curve(
        v_epochs,
        v_acc,
        "Epoch",
        "Validation accuracy",
        "Validation Accuracy vs Epoch",
        os.path.join(out_dir, "val_accuracy.png"),
    )

def _plot_single_curve(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _plot_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        include_values=True, cmap="Blues", ax=ax, xticks_rotation=45
    )
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# ------------------------------------------------------------
# 4.  Main routine -------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Fine‑tune transformer(s) on résumé titles.")
    p.add_argument("--data", required=True, help="Path to résumé JSON file")
    p.add_argument("--exclude", nargs="*", default=[], help="Labels to exclude")
    p.add_argument("--models", nargs="*", default=["bert-base-uncased"], help="HF model names")
    p.add_argument("--lr", nargs="*", type=float, default=[2e-5], help="Learning rates")
    p.add_argument("--bs", nargs="*", type=int, default=[8], help="Batch sizes")
    p.add_argument("--epochs", nargs="*", type=int, default=[5], help="Epoch counts")
    p.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--scheduler", default="linear", choices=["linear", "cosine", "constant"], help="Learning rate scheduler type")
    p.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    p.add_argument("--out", default="experiments", help="Root output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs and exit")
    args = p.parse_args()

    # ---- Load data --------------------------------------------
    with open(args.data, encoding="utf-8") as fp:
        raw = json.load(fp)
    raw = [s for s in raw if s.get("label") not in set(args.exclude)]
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Loaded {len(raw)} samples after exclusion.")

    texts = [resume_to_text(s) for s in raw]
    labels = [s["label"] for s in raw]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Detected {len(le.classes_)} unique labels.")

    # ---- Prepare grid ----------------------------------------
    grid = list(
        itertools.product(args.models, args.lr, args.bs, args.epochs)
    )
    print(f"Planning {len(grid)} run(s)…")

    if args.dry_run:
        for g in grid:
            cfg = ExperimentConfig(
                model_name=g[0],
                learning_rate=g[1],
                batch_size=g[2],
                epochs=g[3],
                max_len=args.max_len,
                seed=args.seed,
                warmup_ratio=args.warmup_ratio,
                gradient_accumulation_steps=args.grad_accum,
                lr_scheduler_type=args.scheduler,
                fp16=args.fp16,
            )
            print("  •", cfg.tag())
        return

    # ---- Run experiments -------------------------------------
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    for g in grid:
        cfg = ExperimentConfig(
            model_name=g[0],
            learning_rate=g[1],
            batch_size=g[2],
            epochs=g[3],
            max_len=args.max_len,
            seed=args.seed,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type=args.scheduler,
            fp16=args.fp16,
        )
        run_experiment(cfg, texts, y, le, args.out)

if __name__ == "__main__":
    main()

"""
Simple Resume Title Classification Model
======================================
This module provides functions for training transformer models on resume title classification.
"""



import os
import json
import torch
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

@dataclass
class ExperimentConfig:
    """Configuration for a single training run."""
    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 4
    max_len: int = 256
    seed: int = 42

def resume_to_text(sample: dict) -> str:
    """Convert resume data to text format."""
    parts = []
    
    # Add education
    for ed in sample.get("education", []):
        parts.append(f"{ed.get('degree','')} {ed.get('field','')} {ed.get('institution','')}")
    
    # Add job history (skip current job)
    jobs = sample.get("job_history", [])
    if jobs and jobs[0].get("end_date") is None:
        jobs = jobs[1:]
    for job in jobs:
        parts.append(f"{job.get('title','')} {job.get('company','')}")
    
    # Add skills
    parts.append(" ".join(sample.get("skills", [])))
    
    return " ".join(parts)

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    
    # Calculate accuracy
    acc = (preds == labels).mean()
    
    # Calculate precision, recall, and F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def run_experiment_with_details(cfg: ExperimentConfig, texts, labels, le: LabelEncoder, out_dir: str, checkpoint_path: str = None):
    """Run a single experiment and return detailed results."""
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name if not checkpoint_path else checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name if not checkpoint_path else checkpoint_path, 
        num_labels=len(le.classes_)
    )
    
    # Prepare datasets
    print("Preparing datasets...")
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=cfg.seed
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=cfg.seed
    )
    
    # Create datasets
    datasets = DatasetDict({
        "train": Dataset.from_dict({"text": X_train, "label": y_train}),
        "valid": Dataset.from_dict({"text": X_valid, "label": y_valid}),
        "test": Dataset.from_dict({"text": X_test, "label": y_test})
    })
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_len
        )
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,

        # ▶ New names in ≥ 4.51
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        logging_steps=10,
        save_steps=100,

        save_total_limit=1,
        overwrite_output_dir=True,
        remove_unused_columns=True,
        report_to="none",
        seed=cfg.seed,
    )
    
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("\nStarting training...")
    print(f"Training on {len(tokenized_datasets['train'])} samples")
    print(f"Validating on {len(tokenized_datasets['valid'])} samples")
    print(f"Will test on {len(tokenized_datasets['test'])} samples")
    
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(tokenized_datasets["test"])
    predictions = trainer.predict(tokenized_datasets["test"])
    
    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(os.path.join(out_dir, "model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    
    return trainer, metrics, predictions
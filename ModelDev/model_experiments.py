#!/usr/bin/env python3
"""
Resume Title Classification Experiments
======================================
This script runs a predefined set of experiments with various models and hyperparameters
to find the best model for resume title classification.

Models tested:
- BERT (base, uncased)
- DistilBERT (base, uncased)
- RoBERTa (base)
- ALBERT (base-v2)
- ELECTRA (base discriminator)

Each model is tested with optimized hyperparameters.
"""

import os
import sys
import json
import inspect
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

# Import from model1.py directly rather than using subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import model1

# Define the models and hyperparameters to test
EXPERIMENTS = [
    # BERT models
    {
        "name": "BERT Base",
        "model": "bert-base-uncased",
        "lr": 2e-5,
        "bs": 8,
        "epochs": 4,
        "max_len": 256,
        "warmup_ratio": 0.1,
    },
    {
        "name": "BERT Base (higher lr)",
        "model": "bert-base-uncased",
        "lr": 5e-5,
        "bs": 8,
        "epochs": 4,
        "max_len": 256,
        "warmup_ratio": 0.1,
    },
    
    # DistilBERT
    {
        "name": "DistilBERT",
        "model": "distilbert-base-uncased",
        "lr": 5e-5,
        "bs": 16,
        "epochs": 5,
        "max_len": 256,
        "warmup_ratio": 0.1,
    },
    
    # RoBERTa 
    {
        "name": "RoBERTa",
        "model": "roberta-base",
        "lr": 2e-5,
        "bs": 8,
        "epochs": 4,
        "max_len": 256,
        "warmup_ratio": 0.1,
    },
    
    # ALBERT
    {
        "name": "ALBERT",
        "model": "albert-base-v2",
        "lr": 1e-5,
        "bs": 8,
        "epochs": 5,
        "max_len": 256,
        "warmup_ratio": 0.1,
    },
    
    # ELECTRA
    {
        "name": "ELECTRA",
        "model": "google/electra-base-discriminator",
        "lr": 3e-5,
        "bs": 16,
        "epochs": 4,
        "max_len": 256,
        "warmup_ratio": 0.1,
    },
]

def extract_training_metrics(trainer):
    """Extract training metrics from trainer logs."""
    logs = trainer.state.log_history
    
    # Extract loss and epoch values from logs
    loss_data = []
    for log in logs:
        if "loss" in log and "epoch" in log:
            loss_data.append({
                "epoch": log["epoch"],
                "loss": log["loss"],
                "learning_rate": log.get("learning_rate", 0),
            })
    
    # Extract evaluation metrics
    eval_data = []
    for log in logs:
        if "eval_loss" in log and "epoch" in log:
            eval_data.append({
                "epoch": log["epoch"],
                "eval_loss": log["eval_loss"],
                "eval_accuracy": log.get("eval_accuracy", 0),
                "eval_precision": log.get("eval_precision", 0),
                "eval_recall": log.get("eval_recall", 0),
                "eval_f1": log.get("eval_f1", 0),
            })
    
    return loss_data, eval_data

def plot_training_curves(loss_data, eval_data, output_dir, title_prefix=""):
    """Plot detailed training curves."""
    # Create DataFrame from the data
    loss_df = pd.DataFrame(loss_data)
    eval_df = pd.DataFrame(eval_data) if eval_data else None
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training loss
    axes[0].plot(loss_df["epoch"], loss_df["loss"], marker='o', linestyle='-', color='blue')
    axes[0].set_title(f"{title_prefix}Training Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate if available
    if "learning_rate" in loss_df.columns:
        ax2 = axes[0].twinx()
        ax2.plot(loss_df["epoch"], loss_df["learning_rate"], marker='x', linestyle='--', color='red')
        ax2.set_ylabel("Learning Rate", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_yscale('log')
    
    # Plot evaluation metrics if available
    if eval_df is not None and not eval_df.empty:
        metrics_to_plot = [col for col in ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"] 
                          if col in eval_df.columns]
        
        for metric in metrics_to_plot:
            axes[1].plot(eval_df["epoch"], eval_df[metric], marker='o', linestyle='-', label=metric.replace("eval_", ""))
        
        axes[1].set_title(f"{title_prefix}Evaluation Metrics vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[1].set_title("No Evaluation Data Available")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title_prefix.replace(' ', '_')}training_curves.png"), dpi=200)
    plt.close(fig)

def generate_class_performance_report(y_true, y_pred, label_names, output_dir, prefix=""):
    """Generate detailed per-class performance report."""
    # Generate classification report and convert to dataframe
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    report_df.to_csv(os.path.join(output_dir, f"{prefix}class_performance.csv"))
    
    # Create a bar chart for per-class performance
    classes = [label_names[i] for i in range(len(label_names)) if i in set(y_true)]
    precision = [report[label]["precision"] for label in classes]
    recall = [report[label]["recall"] for label in classes]
    f1 = [report[label]["f1-score"] for label in classes]
    support = [report[label]["support"] for label in classes]
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1)[::-1]
    classes = [classes[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    f1 = [f1[i] for i in sorted_indices]
    support = [support[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(classes))
    width = 0.2
    
    # Compute max support for sizing
    max_support = max(support)
    normalized_support = [s/max_support*0.8 for s in support]
    
    ax.bar(x - width*1.5, precision, width, label='Precision')
    ax.bar(x - width/2, recall, width, label='Recall')
    ax.bar(x + width/2, f1, width, label='F1')
    ax.bar(x + width*1.5, normalized_support, width, label='Support (normalized)', color='gray', alpha=0.5)
    
    ax.set_ylabel('Score')
    ax.set_title(f'{prefix}Performance by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    
    # Add support values as text
    for i, v in enumerate(support):
        ax.text(i + width*1.5, normalized_support[i] + 0.02, str(v), 
                color='black', fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}class_performance.png"), dpi=200)
    plt.close(fig)
    
    return report_df

def generate_confusion_matrix(y_true, y_pred, label_names, output_dir, prefix=""):
    """Generate and save a confusion matrix."""
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=label_names
    )
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f'{prefix}Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}confusion_matrix.png"), dpi=200)
    plt.close()

def compare_models_detailed(results, class_reports, output_dir):
    """Generate detailed model comparison visualizations."""
    # Extract metrics for comparison
    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for name, metrics in results.items():
        if metrics is not None and "error" not in metrics:
            models.append(name)
            accuracies.append(metrics.get("eval_accuracy", 0))
            precisions.append(metrics.get("eval_precision", 0))
            recalls.append(metrics.get("eval_recall", 0))
            f1s.append(metrics.get("eval_f1", 0))
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    models = [models[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1s = [f1s[i] for i in sorted_indices]
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1s
    })
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(models))
    width = 0.2
    
    ax.bar([i - width*1.5 for i in x], accuracies, width, label='Accuracy')
    ax.bar([i - width*0.5 for i in x], precisions, width, label='Precision')
    ax.bar([i + width*0.5 for i in x], recalls, width, label='Recall')
    ax.bar([i + width*1.5 for i in x], f1s, width, label='F1')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=200)
    plt.close(fig)
    
    # Compare per-class performance across models
    if class_reports:
        # Collect f1-scores for each class across models
        class_scores = {}
        for model_name, report_df in class_reports.items():
            for idx, row in report_df.iterrows():
                if idx not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
                    if idx not in class_scores:
                        class_scores[idx] = []
                    class_scores[idx].append((model_name, row['f1-score']))
        
        # Select top N classes (by support) to avoid overcrowding the plot
        top_n = min(10, len(class_scores))
        top_classes = sorted(class_scores.keys(), 
                            key=lambda c: sum(score for _, score in class_scores[c]),
                            reverse=True)[:top_n]
        
        # Create class comparison plot
        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            class_f1s = []
            for class_name in top_classes:
                model_scores = dict(class_scores[class_name])
                class_f1s.append(model_scores.get(model_name, 0))
            
            ax.bar([j + i*bar_width - 0.4 + bar_width/2 for j in range(len(top_classes))], 
                  class_f1s, bar_width, label=model_name)
        
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison')
        ax.set_xticks(range(len(top_classes)))
        ax.set_xticklabels(top_classes, rotation=45, ha='right')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "class_comparison.png"), dpi=200)
        plt.close(fig)

def run_experiments(data_path: str, exclude_labels: List[str], output_dir: str, run_specific: str = None):
    """Run all the defined experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{output_dir}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create a summary file
    with open(os.path.join(exp_dir, "experiment_summary.txt"), "w") as f:
        f.write(f"Resume Title Classification Experiments\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Excluded labels: {', '.join(exclude_labels) if exclude_labels else 'None'}\n\n")
        f.write("Planned experiments:\n")
        for i, exp in enumerate(EXPERIMENTS, 1):
            f.write(f"{i}. {exp['name']} - {exp['model']}, "
                   f"lr={exp['lr']}, bs={exp['bs']}, epochs={exp['epochs']}\n")
    
    # Determine which experiments to run
    experiments_to_run = []
    if run_specific is not None:
        try:
            indices = [int(idx)-1 for idx in run_specific.split(",")]
            experiments_to_run = [EXPERIMENTS[idx] for idx in indices if 0 <= idx < len(EXPERIMENTS)]
        except:
            print(f"Invalid experiment selection '{run_specific}'. Running all experiments.")
            experiments_to_run = EXPERIMENTS
    else:
        experiments_to_run = EXPERIMENTS
    
    print(f"Running {len(experiments_to_run)} experiments...")
    
    # Load data once
    print("Loading data...")
    with open(data_path, encoding="utf-8") as fp:
        raw = json.load(fp)
    raw = [s for s in raw if s.get("label") not in set(exclude_labels)]
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Loaded {len(raw)} samples after exclusion.")

    texts = [model1.resume_to_text(s) for s in raw]
    labels = [s["label"] for s in raw]

    le = model1.LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Detected {len(le.classes_)} unique labels.")
    
    # Run each experiment
    results = {}
    class_reports = {}
    trainers = {}
    
    for i, exp in enumerate(experiments_to_run, 1):
        exp_name = exp["name"]
        exp_dir_name = os.path.join(exp_dir, f"{i}_{exp_name.replace(' ', '_')}")
        print(f"\n[{i}/{len(experiments_to_run)}] Running experiment: {exp_name} ({exp['model']})")
        
        # Create experiment config
        cfg = model1.ExperimentConfig(
            model_name=exp["model"],
            learning_rate=exp["lr"],
            batch_size=exp["bs"],
            epochs=exp["epochs"],
            max_len=exp["max_len"],
            warmup_ratio=exp["warmup_ratio"],
        )
        
        # Run experiment
        try:
            print(f"Starting experiment with {cfg.model_name}, lr={cfg.learning_rate}, bs={cfg.batch_size}, epochs={cfg.epochs}")
            # Create experiment-specific directory
            os.makedirs(exp_dir_name, exist_ok=True)
            
            # Keep track of the trainer for detailed metrics
            trainer, metrics, predictions = model1.run_experiment_with_details(cfg, texts, y, le, exp_dir_name)
            trainers[exp_name] = trainer
            results[exp_name] = metrics
            
            # Generate detailed class performance report
            y_true = predictions.label_ids
            y_pred = predictions.predictions.argmax(axis=1)
            report_df = generate_class_performance_report(
                y_true, y_pred, le.classes_, exp_dir_name, f"{exp_name}_"
            )
            class_reports[exp_name] = report_df
            
            # Generate confusion matrix
            generate_confusion_matrix(
                y_true, y_pred, le.classes_, exp_dir_name, f"{exp_name}_"
            )
            
            # Plot detailed training curves
            loss_data, eval_data = extract_training_metrics(trainer)
            plot_training_curves(loss_data, eval_data, exp_dir_name, f"{exp_name}_")
            
            print(f"✅ Experiment {i} completed successfully.")
        except Exception as e:
            print(f"❌ Error running experiment {i}: {str(e)}")
            results[exp_name] = {"error": str(e)}
    
    # Create comparison report
    with open(os.path.join(exp_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Compare models
    if len(results) > 1:
        compare_models_detailed(results, class_reports, exp_dir)
    
    # Create summary report
    with open(os.path.join(exp_dir, "experiment_summary.txt"), "a") as f:
        f.write("\n\nExperiment Results:\n")
        for name, metrics in results.items():
            if metrics is None:
                accuracy = "FAILED - No results"
            else:
                accuracy = metrics.get("eval_accuracy", 0) if "error" not in metrics else "FAILED"
            f.write(f"{name}: Accuracy = {accuracy}\n")
        f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nAll experiments completed. Results saved to {exp_dir}")
    
    # Show best model
    best_model = None
    best_acc = -1
    for name, metrics in results.items():
        if metrics is not None and "error" not in metrics and metrics.get("eval_accuracy", 0) > best_acc:
            best_acc = metrics.get("eval_accuracy", 0)
            best_model = name
    
    if best_model:
        print(f"\n🏆 Best model: {best_model} with accuracy {best_acc:.4f}")
    else:
        print("\n❌ No successful experiments")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run resume title classification experiments")
    parser.add_argument("--data", default="ModelDev/Files/data_merged.json", 
                        help="Path to resume JSON data file")
    parser.add_argument("--exclude", nargs="*", default=["Security Engineer", 
                        "Application Security Engineer", "Frontend software engineer"],
                        help="Labels to exclude")
    parser.add_argument("--output", default="model_experiments",
                        help="Output directory for experiment results")
    parser.add_argument("--run", default=None, 
                        help="Run specific experiments by number (e.g. '1,3,5')")
    args = parser.parse_args()
    
    run_experiments(args.data, args.exclude, args.output, args.run) 
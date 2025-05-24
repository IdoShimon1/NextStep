#!/usr/bin/env python3
"""
Simple Resume Title Classification Experiments
============================================
This script runs experiments with different models to classify resume titles.
It tests various transformer models including BERT, DistilBERT, RoBERTa, ALBERT, ELECTRA, DeBERTa, and XLNet.
"""
import pickle, shutil
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Import the model training code
import model1

# Define the models to test
MODELS = [
    {
        "name": "BERT Base",
        "model": "bert-base-uncased",
        "lr": 2e-5,
        "bs": 8,
        "epochs": 8
    },
    {
        "name": "BERT Base (higher lr)",
        "model": "bert-base-uncased",
        "lr": 5e-5,
        "bs": 8,
        "epochs": 8
    },
    {
        "name": "DistilBERT",
        "model": "distilbert-base-uncased",
        "lr": 5e-5,
        "bs": 16,
        "epochs": 8
    },
    {
        "name": "RoBERTa",
        "model": "roberta-base",
        "lr": 2e-5,
        "bs": 8,
        "epochs": 8
    },
    {
        "name": "ALBERT",
        "model": "albert-base-v2",
        "lr": 1e-5,
        "bs": 8,
        "epochs": 5
    },
    {
        "name": "ELECTRA",
        "model": "google/electra-base-discriminator",
        "lr": 3e-5,
        "bs": 16,
        "epochs": 4
    },
    {
        "name": "DeBERTa",
        "model": "microsoft/deberta-base",
        "lr": 2e-5,
        "bs": 8,
        "epochs": 4
    },
    {
        "name": "XLNet",
        "model": "xlnet-base-cased",
        "lr": 2e-5,
        "bs": 8,
        "epochs": 4
    }
]

def load_data(data_path, exclude_labels=None):
    """Load and prepare the resume data."""
    with open(data_path, encoding="utf-8") as fp:
        raw = json.load(fp)
    
    if exclude_labels:
        raw = [s for s in raw if s.get("label") not in set(exclude_labels)]
    
    texts = [model1.resume_to_text(s) for s in raw]
    labels = [s["label"] for s in raw]
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"Loaded {len(raw)} samples with {len(le.classes_)} unique labels")
    return texts, y, le

def plot_training_results(trainer, output_dir, model_name):
    """Plot enhanced training loss, accuracy, and other metrics curves with per-epoch metrics, including validation loss."""
    logs = trainer.state.log_history
    train_data = []
    eval_data = []
    
    # Extract training and evaluation data
    step_to_epoch = {}  # Map from step to epoch
    
    # First pass: collect step to epoch mapping from training logs
    for log in logs:
        if "loss" in log and "epoch" in log and "step" in log:
            step_to_epoch[log["step"]] = log["epoch"]
    
    # Second pass: collect training and evaluation data
    for log in logs:
        # Training data
        if "loss" in log and "epoch" in log and "step" in log:
            train_data.append({
                "step": log["step"],
                "epoch": log["epoch"],
                "loss": log["loss"]
            })
        
        # Evaluation data - include all validation metrics
        if "eval_loss" in log:
            # Try to get the epoch from step_to_epoch mapping
            step = log.get("step", None)
            epoch = log.get("epoch", None)
            
            if epoch is None and step is not None and step in step_to_epoch:
                epoch = step_to_epoch[step]
            
            # If we still don't have an epoch, estimate it based on step and total steps
            if epoch is None and step is not None:
                # Estimate epoch based on step and total steps (approximate)
                total_steps = max(step_to_epoch.keys()) if step_to_epoch else 0
                if total_steps > 0:
                    epoch = (step / total_steps) * max(step_to_epoch.values())
            
            eval_data.append({
                "step": step,
                "epoch": epoch,
                "accuracy": log.get("eval_accuracy", 0),
                "f1": log.get("eval_f1", 0),
                "precision": log.get("eval_precision", 0),
                "recall": log.get("eval_recall", 0),
                "val_loss": log.get("eval_loss", 0)
            })

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # Training loss vs step/epoch
    if train_data:
        train_df = pd.DataFrame(train_data)
        
        # Plot against epoch if available, otherwise use step
        x_col = "epoch" if "epoch" in train_df.columns and not train_df["epoch"].isna().all() else "step"
        
        axes[0, 0].plot(train_df[x_col], train_df["loss"], marker='o', color='#FF5733', linewidth=2)
        axes[0, 0].set_title(f"Training Loss vs {x_col.capitalize()}", fontsize=16, fontweight='bold')
        axes[0, 0].set_xlabel(x_col.capitalize(), fontsize=14)
        axes[0, 0].set_ylabel("Loss", fontsize=14)
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for key points
        for i in range(0, len(train_df), max(1, len(train_df) // 10)):  # Annotate ~10 points
            row = train_df.iloc[i]
            axes[0, 0].annotate(f"{row['loss']:.4f}", 
                              (row[x_col], row['loss']), 
                              textcoords="offset points", 
                              xytext=(0,10), 
                              ha='center', 
                              fontsize=10)
    
    # Validation metrics vs step/epoch
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        
        # Plot against epoch if available, otherwise use step
        x_col = "epoch" if "epoch" in eval_df.columns and not eval_df["epoch"].isna().all() else "step"
        
        # Plot all validation metrics
        axes[0, 1].plot(eval_df[x_col], eval_df["accuracy"], marker='o', color='#33A1FF', linewidth=2, label='Accuracy')
        axes[0, 1].plot(eval_df[x_col], eval_df["f1"], marker='s', color='#33FF57', linewidth=2, label='F1')
        axes[0, 1].plot(eval_df[x_col], eval_df["precision"], marker='^', color='#D133FF', linewidth=2, label='Precision')
        axes[0, 1].plot(eval_df[x_col], eval_df["recall"], marker='d', color='#FFD133', linewidth=2, label='Recall')
        
        axes[0, 1].set_title(f"Validation Metrics vs {x_col.capitalize()}", fontsize=16, fontweight='bold')
        axes[0, 1].set_xlabel(x_col.capitalize(), fontsize=14)
        axes[0, 1].set_ylabel("Score", fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].set_ylim(0, 1.0)
        
        # Validation loss
        if "val_loss" in eval_df.columns and not eval_df["val_loss"].isna().all():
            axes[0, 2].plot(eval_df[x_col], eval_df["val_loss"], marker='o', color='#FF33A1', linewidth=2)
            axes[0, 2].set_title(f"Validation Loss vs {x_col.capitalize()}", fontsize=16, fontweight='bold')
            axes[0, 2].set_xlabel(x_col.capitalize(), fontsize=14)
            axes[0, 2].set_ylabel("Loss", fontsize=14)
            axes[0, 2].grid(True, linestyle='--', alpha=0.7)
        
        # Final metrics bar chart (still using the last evaluation)
        last_eval = eval_df.iloc[-1]
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        values = [last_eval[m] for m in metrics]
        colors = ['#33A1FF', '#33FF57', '#D133FF', '#FFD133']
        
        axes[1, 0].bar(metrics, values, color=colors)
        axes[1, 0].set_title("Final Metrics", fontsize=16, fontweight='bold')
        axes[1, 0].set_ylim(0, 1.0)
        axes[1, 0].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for i, v in enumerate(values):
            axes[1, 0].annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
        
        # Add validation metrics progression chart
        if len(eval_df) > 1:  # Only if we have multiple evaluation points
            axes[1, 1].set_title("Validation Metrics Progression", fontsize=16, fontweight='bold')
            
            for metric, color in zip(metrics, colors):
                axes[1, 1].plot(eval_df[x_col], eval_df[metric], marker='o', color=color, linewidth=2, label=metric.capitalize())
            
            axes[1, 1].set_xlabel(x_col.capitalize(), fontsize=14)
            axes[1, 1].set_ylabel("Score", fontsize=14)
            axes[1, 1].legend()
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)
            axes[1, 1].set_ylim(0, 1.0)
    
    # Hide unused subplot if no val_loss
    if not (eval_data and "val_loss" in pd.DataFrame(eval_data).columns and not pd.DataFrame(eval_data)["val_loss"].isna().all()):
        axes[0, 2].axis('off')
    
    # Hide unused subplot if we don't have enough eval data
    if not eval_data or len(pd.DataFrame(eval_data)) <= 1:
        axes[1, 1].axis('off')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle(f"{model_name} Training Performance", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"{model_name}_training_curves.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, output_dir, model_name):
    """Plot enhanced confusion matrix for model predictions."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots - absolute and normalized values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12))
    
    # Plot absolute numbers
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp1.plot(cmap='Blues', ax=ax1, xticks_rotation=45, values_format='d', colorbar=True)
    ax1.set_title(f"{model_name} Confusion Matrix (Counts)", fontsize=18)
    
    # Plot normalized values (percentage)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp2.plot(cmap='Blues', ax=ax2, xticks_rotation=45, values_format='.2f', colorbar=True)
    ax2.set_title(f"{model_name} Confusion Matrix (Normalized)", fontsize=18)
    
    # Improve axes labels
    for ax in [ax1, ax2]:
        ax.set_xlabel("Predicted Label", fontsize=14, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=14, fontweight='bold')
        
        # Make axis tick labels larger and bold
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Adjust label positions for better readability
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    
    # Create additional visualization for model performance metrics
    y_pred_probs = np.zeros((len(y_true), len(labels)))
    for i, pred in enumerate(y_pred):
        y_pred_probs[i, pred] = 1
    
    # Create class-wise metrics
    class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Extract metrics for each class
    classes = []
    precision = []
    recall = []
    f1 = []
    support = []
    
    for cls in labels:
        if cls in class_report:
            classes.append(cls)
            precision.append(class_report[cls]['precision'])
            recall.append(class_report[cls]['recall'])
            f1.append(class_report[cls]['f1-score'])
            support.append(class_report[cls]['support'])
    
    # Plot class-wise metrics
    fig, ax = plt.subplots(figsize=(18, 8))
    
    x = np.arange(len(classes))
    width = 0.2
    
    ax.bar(x - width, precision, width, label='Precision', color='#33A1FF')
    ax.bar(x, recall, width, label='Recall', color='#FF5733')
    ax.bar(x + width, f1, width, label='F1', color='#33FF57')
    
    ax.set_xlabel('Classes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} Class-wise Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add support counts
    for i, v in enumerate(support):
        ax.annotate(f"n={v}", xy=(i, 0.05), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_class_metrics.png"), dpi=300)
    plt.close()

    # Support per class
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(classes, support, color='#FFB347')
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
    ax.set_title(f"{model_name} Test Set Support per Class", fontsize=16, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_class_support.png"), dpi=300)
    plt.close()

def plot_model_comparison(results, output_dir):
    """Plot a comparison graph between all models."""
    # Filter out models with errors
    valid_results = {name: metrics for name, metrics in results.items() if "error" not in metrics}
    
    if not valid_results:
        print("❌ No valid models to compare")
        return
    
    # Prepare data for plotting
    model_names = list(valid_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_colors = ['#33A1FF', '#D133FF', '#FFD133', '#33FF57']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # Bar chart - all metrics for all models
    x = np.arange(len(model_names))
    width = 0.2
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [valid_results[name][metric] for name in model_names]
        ax1.bar(x + (i - 1.5) * width, values, width, label=metric.capitalize(), color=metric_colors[i])
    
    # Add labels and styling
    ax1.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('Model Comparison - All Metrics', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add value annotations
    for i, metric in enumerate(metrics):
        values = [valid_results[name][metric] for name in model_names]
        for j, v in enumerate(values):
            ax1.annotate(f"{v:.3f}", 
                        xy=(j + (i - 1.5) * width, v), 
                        xytext=(0, 3),
                        textcoords="offset points", 
                        ha='center', 
                        fontsize=9)
    
    # Radar chart - model comparison
    # Create radar chart
    categories = metrics
    N = len(categories)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the radar chart
    ax2 = plt.subplot(2, 1, 2, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=14)
    
    # Draw the y-axis labels (0.2 to 1.0)
    ax2.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
               color="grey", size=12)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, (name, metrics_dict) in enumerate(valid_results.items()):
        values = [metrics_dict[m] for m in categories]
        values += values[:1]  # Close the loop
        
        # Choose a color from a colormap
        color = plt.cm.tab10(i / len(valid_results))
        
        # Plot values
        ax2.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=color)
        ax2.fill(angles, values, color=color, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax2.set_title('Model Comparison - Radar Chart', fontsize=16, fontweight='bold', pad=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_experiments(data_path="ModelDev/Files/data_merged.json", 
                   output_dir="experiments",
                   exclude_labels=["Security Engineer", "Frontend software engineer"],
                   checkpoint_path=None):
    """Run experiments with different models."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{output_dir}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create experiment summary file
    summary_path = os.path.join(exp_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Resume Title Classification Experiments\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Excluded labels: {', '.join(exclude_labels)}\n\n")
        
        f.write("Planned experiments:\n")
        for i, model_config in enumerate(MODELS, 1):
            f.write(f"{i}. {model_config['name']} - {model_config['model']}, "
                   f"lr={model_config['lr']}, bs={model_config['bs']}, "
                   f"epochs={model_config['epochs']}\n")
        f.write("\n")
    
    print("\n🚀 Starting Resume Title Classification Experiments")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading and preparing data...")
    texts, y, le = load_data(data_path, exclude_labels)
    label_path = os.path.join(exp_dir, "label_encoder.pkl")
    with open(label_path, "wb") as fp:
        pickle.dump(le, fp)
    print(f"✓ Saved label encoder → {label_path}")
    print(f"✓ Loaded {len(texts)} resumes")
    print(f"✓ Found {len(le.classes_)} unique job titles")
    print(f"✓ Job titles: {', '.join(le.classes_)}")
    
    # Run experiments
    results = {}
    print("\n🤖 Starting model training...")
    print("=" * 50)
    
    for model_config in MODELS:
        model_name = model_config["name"]
        print(f"\n📌 Training {model_name}")
        print("-" * 30)
        print(f"Model: {model_config['model']}")
        print(f"Learning rate: {model_config['lr']}")
        print(f"Batch size: {model_config['bs']}")
        print(f"Epochs: {model_config['epochs']}")
        
        # Create model directory
        model_dir = os.path.join(exp_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(label_path, os.path.join(model_dir, "label_encoder.pkl"))   # NEW

        try:
            # Configure and run experiment
            cfg = model1.ExperimentConfig(
                model_name=model_config["model"],
                learning_rate=model_config["lr"],
                batch_size=model_config["bs"],
                epochs=model_config["epochs"]
            )
            
            print("\n⏳ Training in progress...")
            # Run training
            trainer, metrics, predictions = model1.run_experiment_with_details(
                cfg, texts, y, le, model_dir, checkpoint_path
            )
            
            # Plot results
            print("📈 Generating plots...")
            plot_training_results(trainer, model_dir, model_name)
            
            # Generate confusion matrix
            print("📊 Creating confusion matrix...")
            y_true = predictions.label_ids
            y_pred = predictions.predictions.argmax(axis=1)
            plot_confusion_matrix(y_true, y_pred, le.classes_, model_dir, model_name)
            
            # Save results
            results[model_name] = {
                "accuracy": metrics.get("eval_accuracy", 0),
                "precision": metrics.get("eval_precision", 0),
                "recall": metrics.get("eval_recall", 0),
                "f1": metrics.get("eval_f1", 0)
            }
            
            print(f"\n✅ {model_name} completed successfully")
            print(f"   Accuracy: {results[model_name]['accuracy']:.4f}")
            print(f"   F1 Score: {results[model_name]['f1']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error with {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # Save results
    print("\n💾 Saving results...")
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Update experiment summary with results
    with open(summary_path, "a") as f:
        f.write("\nExperiment Results:\n")
        for model_name, metrics in results.items():
            if "error" not in metrics:
                f.write(f"{model_name}: Accuracy = {metrics['accuracy']}\n")
            else:
                f.write(f"{model_name}: Error - {metrics['error']}\n")
        f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create model comparison plot
    print("\n📊 Generating model comparison plot...")
    plot_model_comparison(results, exp_dir)
    
    # Print best model
    best_model = None
    best_acc = -1
    for name, metrics in results.items():
        if "error" not in metrics and metrics.get("accuracy", 0) > best_acc:
            best_acc = metrics.get("accuracy", 0)
            best_model = name
    
    print("\n🏆 Final Results")
    print("=" * 50)
    if best_model:
        print(f"Best model: {best_model}")
        print(f"Best accuracy: {best_acc:.4f}")
        print(f"\nDetailed metrics for {best_model}:")
        for metric, value in results[best_model].items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("❌ No successful experiments")
    
    print(f"\n📁 All results saved to: {exp_dir}")
    print(f"📊 Model comparison chart saved to: {os.path.join(exp_dir, 'model_comparison.png')}")

if __name__ == "__main__":
    run_experiments() 
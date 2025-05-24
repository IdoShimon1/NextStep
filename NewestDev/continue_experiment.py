#!/usr/bin/env python3
"""
Continue Resume Title Classification Experiment
============================================
This script continues a previous experiment from a checkpoint.
"""

from model_experiments import run_experiments

if __name__ == "__main__":
    # Path to the RoBERTa checkpoint
    checkpoint_path = "experiments_20250507_221644/RoBERTa/checkpoint-400"
    
    # Run the experiment with the checkpoint
    run_experiments(
        data_path="ModelDev/Files/data_merged.json",
        output_dir="experiments_continued",
        exclude_labels=["Security Engineer", "Frontend software engineer"],
        checkpoint_path=checkpoint_path
    ) 
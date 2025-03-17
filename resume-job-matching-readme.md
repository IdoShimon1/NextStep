# Resume-to-Job Matching Project

## Submitters
- Ido Shimon 215885351
- Nicole Goihman 215871583
- Yana Prokhorov 323518621
- Yael Itzkovitch 211747852

## Project's GitHub
https://github.com/IdoShimon1/NextStep

## Overview
This project implements an automated system that analyzes resumes and matches them to suitable job positions based on skills, education, and experience. The system uses machine learning models to cluster job titles, classify resumes into job clusters, and predict match percentages between resumes and job postings.

## Features
- **Resume Parsing**: Extracts structured information (job titles, skills, education, experience) from resume text
- **Job Clustering**: Groups similar job titles based on their descriptions and requirements
- **Job Matching**: Predicts how well a resume matches different job positions using a regression model
- **End-to-End Pipeline**: Integrates all components for seamless resume-to-job matching

## How It Works
The project follows these steps:
1. **Data Enrichment**: Job title data is enriched with education and experience requirements
2. **Synthetic Data Generation**: Creates synthetic resume-job pairs to train the matching algorithm
3. **Job Clustering**: Uses BERT embeddings to cluster similar job titles
4. **Resume Classification**: Trains a Random Forest classifier to determine which job cluster a resume belongs to
5. **Match Prediction**: Uses a Linear Regression model to predict match percentages between resumes and jobs

## Project Structure
- **Data Enrichment**: Processes job titles to add education and experience requirements
- **Synthetic Data Generation**: Creates training data for the matching algorithm
- **Clustering Model**: Groups similar job titles using BERT embeddings
- **Classification Model**: Determines which job cluster a resume belongs to
- **Regression Model**: Predicts match percentages between resumes and jobs
- **Final Integration**: Connects all models for end-to-end prediction

## Getting Started

### Prerequisites
The notebook requires the following Python packages:
- gdown
- sentence_transformers
- langdetect
- transformers
- scikit-learn
- matplotlib
- torch
- pandas
- numpy


### Running the Project
1. Open the notebook in Google Colab or Jupyter Notebook
2. Run the installation cell to install all required packages
3. Run the data download cell to download the required data files:
   ```python
   import gdown
   !mkdir /content/Data
   
   !gdown --id 1WkuOsq5WOEnYTZdURtM8E3QI-jZeiLEE -O /content/Data/job_titles.csv
   !gdown --id 1MBr12BNwMDJMZdwU8a7W6mlldI6tcc9G -O /content/Data/Enriched_and_cleaned_data.csv
   !gdown --id 1Sh7r4R5-XuCyIvmfUTzgswr4NW9tsIMJ -O /content/Data/clusters_summary1.csv
   !gdown --id 1eN18qNhQwlsAVQDiZRYM0WgEtLI6Igwm -O /content/Data/random_forest_classifier.pkl
   !gdown --id 19zejr7wGT0LyhEBqnk6ot606tvOkeBpx -O /content/Data/linear_regression_model.pkl
   !gdown --id 1txqE9CxJsNPCKJb2GPbB0RoZehbdqCCR -O /content/Data/Random_Resumes.csv
   ```
4. Run the final integration cell to execute the complete job matching pipeline

## Testing with Your Own Resume

To test the model with your own resume:
(you can just run all cells and it will work, otherwise:)

1. Make sure you've run all the setup cells to download the necessary model files and data
2. Have your resume saved as a plain text (.txt) file
3. When you run the final integration cell, a file upload dialog will appear automatically
4. Click on the "Choose Files" button and select your resume text file
5. After uploading, the model will:
   - Extract information from your resume (job titles, skills, education, experience)
   - Classify your resume into a job cluster
   - Show you which jobs in that cluster match your resume
   - Provide a match percentage for each job

The output will be displayed in the notebook and also saved to a CSV file at `/content/Data/final_predicted_matches.csv`.

## Important Notes
- The resume should be in plain text format (.txt)
- Should run the notebook in gogal colab, because of pre-installed packages
- The model works best with clear, structured resumes that list skills, education, and experience
- The OpenAI API key in the notebook is for demonstration purposes and has usage limits

## Troubleshooting
- If you encounter issues with file uploads, try running the notebook in Google Colab
- If you get an error about missing files, ensure you've run the data download cell successfully

## Output Explanation
The final output includes:
- Extracted resume information (job titles, skills, education, experience)
- The predicted job cluster for your resume
- A list of job titles in that cluster
- Match percentages for each job
- Detailed scores for skills similarity, education match, and experience closeness

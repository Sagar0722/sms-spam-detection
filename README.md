# Task 4: Sms Spam Detection with Machine Learning

## Overview
Spam sms (junk sms) is a type of sms sent to a massive number of users at one time, frequently containing cryptic messages, scams, or phishing content. This project builds an sms spam detector using Python and machine learning to classify sms as **spam** or **ham (not spam)**.

## Dataset
Download the dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Save the file as `spam.csv` in this project folder.

## Project Structure
```
sms_spam_detection/
│
├── sms_spam_detection.py     # Main ML script
├── sms_spam_detection.ipynb  # Jupyter Notebook version
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Steps Covered
1. Load & explore the dataset
2. Data exploration (message length, word count analysis)
3. Text preprocessing (lowercase, remove punctuation/numbers)
4. Train/test split with stratification
5. Model training with TF-IDF vectorization:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - SVM (Support Vector Machine)
6. Model evaluation (accuracy, classification report, confusion matrix)
7. ROC Curves with AUC scores
8. Predict new/custom messages

## How to Run

### Option 1: Python Script
```bash
pip install -r requirements.txt
python email_spam_detection.py
```

### Option 2: Jupyter Notebook
```bash
pip install -r requirements.txt
jupyter notebook email_spam_detection.ipynb
```

## Expected Output
- Console logs with accuracy & classification reports
- `spam_detection_results.png` — 4-panel visualization
- `roc_curves.png` — ROC curves for all models

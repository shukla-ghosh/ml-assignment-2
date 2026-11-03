# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to compare five supervised
classification algorithms for predicting the variety of a dry bean
from its numerical measurements.

The models are Logistic Regression, Decision Tree, K-Nearest
Neighbors, Gaussian Naive Bayes, and Support Vector Machine.

## 2. Dataset Description

Dataset: Dry Bean Dataset

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

Number of instances: 13611
Number of input features: 16
Target column: Class
Number of target classes: 7

The dataset contains numerical measurements of dry beans.
The target represents the bean variety.

## 3. GitHub Repository Link

REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL

## 4. Models Used and Evaluation

The data was split into training and testing sets using an 80:20
split with stratification and random_state=42.

Feature standardization was included in the pipelines for models
that benefit from scaling. All models were trained on the training
set and evaluated on the held-out test set.

AUC was calculated using weighted one-versus-rest multiclass AUC.
Precision, recall, and F1 were calculated using weighted averages.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression    |     0.9207 | 0.9934 |      0.9215 |   0.9207 |     0.9209 | 0.9042 |
| Decision Tree          |     0.892  | 0.9334 |      0.8917 |   0.892  |     0.8916 | 0.8696 |
| KNN                    |     0.9166 | 0.9812 |      0.9174 |   0.9166 |     0.9168 | 0.8992 |
| Naive Bayes            |     0.8979 | 0.9902 |      0.9007 |   0.8979 |     0.8981 | 0.8773 |
| Support Vector Machine |     0.9221 | 0.9937 |      0.9224 |   0.9221 |     0.9222 | 0.9059 |

## 5. Observations

Replace this section with your interpretation of the actual results.

- Logistic Regression: Discuss its measured metrics and errors.
- Decision Tree: Discuss its measured metrics and errors.
- KNN: Discuss its measured metrics and errors.
- Naive Bayes: Discuss its measured metrics and errors.
- Support Vector Machine: Discuss its measured metrics and errors.

Overall winner:
Choose the model based on your results and explain the reason.
Consider all six metrics rather than relying on only one metric.

## 6. Streamlit Application

The app allows a user to upload labeled test data, select a saved
classifier, view evaluation metrics, inspect a classification report,
and view a confusion matrix.

The application loads saved model pipelines and does not retrain
them on the uploaded test data.

Live application:
REPLACE_WITH_YOUR_STREAMLIT_APP_URL

## 7. Project Files

- app.py: Streamlit application
- requirements.txt: Python dependencies
- README.md: Project documentation
- test_data.csv: Labeled held-out test data
- model/: Saved pipelines for the five classifiers
- metadata.json: Feature names, target name, classes, and model files

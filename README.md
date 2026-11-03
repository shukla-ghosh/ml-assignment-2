# Machine Learning Assignment 2
## Comparative Analysis of Classification Models on the Dry Bean Dataset

### 1. Problem Statement

The objective of this assignment is to implement and compare five supervised machine learning classification algorithms on a common dataset. The models are evaluated using multiple performance metrics to identify the most suitable classification algorithm.

The five models implemented are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository

The Dry Bean Dataset contains numerical measurements extracted from images of dry bean grains. The objective is to classify beans into their respective categories based on their physical characteristics.

- Number of features: 16
- Number of instances: More than 13,000
- Problem type: Multiclass classification
- Target variable: Bean class

The dataset satisfies the assignment requirements of having at least 12 features and 500 instances.

### 3. Project Links

**GitHub Repository:**  
https://github.com/shukla-ghosh/ml-assignment-2

**Live Streamlit Application:**  
https://ml-assignment-2-wmc9g2x5ahvkmuvtxr5wpc.streamlit.app/

### 4. Models Implemented

The following five classification algorithms were implemented and evaluated using the same dataset and training/testing split.

| Model | Description |
|---|---|
| Logistic Regression | A linear classification algorithm that estimates class probabilities. |
| Decision Tree | A tree-based model that makes predictions using decision rules. |
| K-Nearest Neighbors | A distance-based algorithm that classifies samples using their nearest neighbors. |
| Gaussian Naive Bayes | A probabilistic classifier based on Bayes' theorem and Gaussian feature distributions. |
| Support Vector Machine | A classification algorithm that identifies an optimal separating hyperplane between classes. |

### 5. Evaluation Metrics

The models were compared using the following six evaluation metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between classes.
- **Precision:** Measures how many predicted positive classifications are correct.
- **Recall:** Measures how many actual positive instances are correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient measures the quality of classification while considering all parts of the confusion matrix.

### 6. Model Performance Comparison

The following table presents the actual evaluation results obtained from the experiment.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

### 7. Observations

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 92.07% and an AUC of 0.9934. It demonstrated strong and balanced classification performance across the evaluation metrics. Its results were very close to those of SVM.

**2. Decision Tree**

Decision Tree achieved an accuracy of 89.20%. It recorded the lowest performance among the five models across the reported evaluation metrics. This suggests that its classification performance was comparatively weaker on this dataset.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 91.66% and an AUC of 0.9812. It showed good classification performance and produced results close to Logistic Regression and SVM.

**4. Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 89.79% and an AUC of 0.9902. Although its AUC was high, its accuracy and other classification metrics were lower than those of Logistic Regression, KNN, and SVM.

**5. Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 92.21%, AUC of 0.9937, precision of 92.24%, recall of 92.21%, F1 score of 92.22%, and MCC of 0.9059. It delivered the best overall performance among all five models.

### 8. Overall Conclusion

Based on the experimental results, **Support Vector Machine (SVM) is the best-performing model** for the Dry Bean classification task.

It achieved the highest values across all six evaluation metrics, indicating strong classification performance and a good balance between precision and recall.

Logistic Regression was a close second, while KNN also demonstrated competitive performance. Decision Tree recorded the lowest overall results among the evaluated models.

Therefore, SVM is selected as the overall winner of this comparative analysis.

### 9. Streamlit Application

A Streamlit web application was developed to make the trained classification models accessible through an interactive interface.

The application provides the following functionalities:

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Evaluate the selected model on the uploaded test data.
- View performance metrics.
- View the confusion matrix and classification report.

The application loads the previously saved models and evaluates the uploaded test data without retraining them.

**Live Application:**  
https://ml-assignment-2-wmc9g2x5ahvkmuvtxr5wpc.streamlit.app/

### 10. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Google Colab
- BITS Virtual Lab
- GitHub

### 11. Repository

The complete project source code, saved models, preprocessing artifacts, test data, and supporting files are maintained in the GitHub repository.

**GitHub Repository:**  
https://github.com/shukla-ghosh/ml-assignment-2

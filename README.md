# Breast Cancer Classification with Logistic Regression

This project implements a binary classifier to predict whether breast cancer is **Benign** or **Malignant** using the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Features
- **Data Preprocessing**: Handles data loading and feature standardization.
- **Machine Learning**: Utilizes Scikit-learn's `LogisticRegression`.
- **Evaluation**: Comprehensive metrics including Accuracy, Precision, Recall, Confusion Matrix, and ROC-AUC.
- **Visualization**: Generates an ROC Curve plot to visualize model performance.

## Getting Started

### Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
pip install scikit-learn pandas matplotlib numpy
```

### Running the Classifier
Run the main script to train the model and see the evaluation metrics:
```bash
python classify_breast_cancer.py
```

## Results
The model achieves high performance on the test set (approx. 97% accuracy).
- **ROC-AUC**: ~0.99
- **Accuracy**: ~97%

## Documentation
- [Sigmoid Function Explanation](sigmoid_explanation.md): A detailed look at the core of Logistic Regression.

## Dataset
This project uses the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) from UCI Machine Learning Repository.

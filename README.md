# Heart Disease Prediction Using Machine Learning

## Overview
This project explores a heart disease dataset to develop predictive models capable of identifying individuals with heart disease. The goal is to compare the performance of two machine learning models—Logistic Regression (LR) and a Deep Neural Network (DNN)—to determine which is better suited for this dataset. Emphasis was placed on recall to minimize false negatives, ensuring potential patients are correctly identified.

## Key Skills
- Machine learning model development and evaluation
- Logistic Regression and Deep Neural Networks
- Data preprocessing: encoding categorical variables, feature scaling
- Exploratory Data Analysis (EDA): histograms, scatter plots, bar charts
- Model evaluation metrics: accuracy, precision, recall, F1-score
- Python programming: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, PyTorch
- Predictive modeling for healthcare datasets
- Train-validation-test splitting and data cleaning

## Dataset
- Source: Kaggle Heart Disease Dataset  
- Samples: 303  
- Attributes: 14 (age, sex, blood pressure, cholesterol, chest pain type, and target variable)  
- Target: 1 = disease, 0 = no disease  
- No missing values, duplicate rows removed  

## Exploratory Data Analysis (EDA)
- Continuous features analyzed for distribution and skewness
- Categorical features encoded and visualized using bar charts
- Target distribution slightly imbalanced towards no disease (0)
- Visualizations included histograms, scatter plots, and bar charts

## Data Preprocessing
- One-hot encoding for multi-class categorical variables
- Binary categorical variables converted to integers
- Continuous features scaled for uniform ranges
- Train-validation-test split: 70%-15%-15%  

## Machine Learning Models

### Logistic Regression (LR)
**Approach:** Linear model chosen for interpretability and strong performance in healthcare datasets.  
**Training & Evaluation:** Trained on the training set and evaluated using accuracy, precision, recall, and F1-score.  
**Results:**  
- Accuracy: 85%  
- Precision: Class 0 = 0.95, Class 1 = 0.75  
- Recall: Class 0 = 0.78, Class 1 = 0.95  
- F1-Score: Class 0 = 0.86, Class 1 = 0.84  

### Deep Neural Network (DNN)
**Approach:** Three fully connected layers with ReLU activation and a sigmoid output layer.  
**Training & Evaluation:** Trained for 50 epochs with Adam optimizer and Binary Cross-Entropy loss. Validation loss plateaued, indicating potential overfitting.  
**Results:**  
- Accuracy: 75%  
- Precision: Class 0 = 0.81, Class 1 = 0.70  
- Recall: Class 0 = 0.74, Class 1 = 0.78  
- F1-Score: Class 0 = 0.77, Class 1 = 0.74  

## Conclusion & Recommendations
- Logistic Regression outperformed the DNN on this dataset, showing better accuracy and balanced precision/recall.  
- LR is recommended for immediate deployment in settings requiring interpretability and reliability.  
- Future work: experiment with advanced architectures, cross-dataset validation, and hyperparameter tuning for DNNs.

## Files Included
- `heart_disease_prediction.ipynb` – Jupyter Notebook with data exploration, preprocessing, and model building  
- `heart_disease_prediction.py` – Python script version of the notebook for execution  





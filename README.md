**Diabetes Prediction: Bayesian Networks vs. Classical Machine Learning**
 **Project Overview**
This project is my MSc Artificial Intelligence dissertation. The objective was to build an interpretable Bayesian Network for predicting diabetes and benchmark its performance against four classical machine learning models.

Using the Pima Indians Diabetes Dataset (768 female patients, 8 clinical features), the study focuses on moving beyond standard accuracy metrics to evaluate model reliability through probability calibration and sensitivity analysis.

**Technologies & Libraries Used****
Language: Python 3.10

Machine Learning: scikit-learn, xgboost, bnlearn, pgmpy

Data Manipulation & Viz: pandas, numpy, matplotlib, seaborn

** Methodology**
Data Cleaning: Handled invalid zero-values across clinical features (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) by imputing them with the feature mean.

Model Benchmarking: Trained and evaluated five models:

Bayesian Network (Expert-designed structure)

Bayesian Network (Data-learned structure using Hill Climbing with K2 scoring)

Logistic Regression

Random Forest

XGBoost

Neural Network (MLP)

Advanced Evaluation: Evaluated models using eight metrics, including probability calibration metrics (Brier Score, Expected Calibration Error, and Log Loss) rather than just accuracy.

** Key Findings**
Domain Knowledge Matters: The expert-designed Bayesian Network outperformed the data-learned structure on all evaluation metrics, demonstrating that domain knowledge acts as an effective regulariser on small datasets.

Accuracy vs. Reliability: While the Neural Network achieved the highest raw accuracy, Logistic Regression was the best-calibrated model. This highlights that accuracy alone is a poor guide for model selection in high-stakes medical predictions.

Key Predictors: Sensitivity and robustness analysis identified Glucose as the dominant predictor, while confirming a non-additive interaction between Glucose and BMI.

** How to Run**
Clone this repository.

Ensure you have the required libraries installed:

bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost bnlearn pgmpy
Open bayesian_diabetes_fixed_(1)_(2).ipynb in Jupyter Notebook or Google Colab.

Run the cells sequentially. The notebook includes full data visualisations (EDA heatmaps, correlation matrices, and model performance graphs).


# Bayesian Networks for Diabetes Prediction

### A Comparative Analysis with Classical Machine Learning Models

MSc Artificial Intelligence dissertation  
University of Huddersfield | 2026  
Author: Keerat Anmol Singh  
Supervisor: Dr Zahid Hussain

[Notebook](Diabetes_Prediction_Bayesian_vs_ML.ipynb) | [Dataset](diabetes.csv) | [GitHub](https://github.com/iamkeerat)

---

This project investigates whether an interpretable Bayesian Network can compete with classical machine learning models for diabetes prediction. It goes beyond accuracy by evaluating probability calibration, comparing expert-designed and data-learned network structures, demonstrating probabilistic inference, and analysing feature sensitivity.

A Bayesian Network represents variables as nodes in a directed acyclic graph (DAG) and encodes dependencies through conditional probability tables (CPTs).

## At a glance

| Item | Detail |
| --- | --- |
| Task | Binary classification of diabetes |
| Dataset | Pima Indians Diabetes Dataset: 768 patients, 8 clinical features |
| Models | Expert Bayesian Network, learned Bayesian Network, Logistic Regression, Random Forest, XGBoost, Neural Network |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, Brier score, Expected Calibration Error (ECE), Log Loss, Structural Hamming Distance (SHD), sensitivity analysis |
| Best accuracy | Neural Network, 77.27% |
| Best calibration | Logistic Regression, Brier score 0.1627 |
| Main finding | The expert Bayesian Network was competitive and interpretable, while the learned DAG overfit the small sample |

## Research questions

| RQ | Question | Summary of finding |
| --- | --- | --- |
| RQ1 | How does a Bayesian Network compare with classical ML models? | The Bayesian Network reached 72.73% accuracy, 4.55 percentage points below the best Neural Network |
| RQ2 | How reliable are the Bayesian Network's probability outputs? | Probability clipping brought Brier score to 0.2230, below the 0.2277 base-rate baseline |
| RQ3 | How does an expert-designed DAG compare with a learned DAG? | The expert DAG outperformed the learned DAG on all metrics and avoided severe overfitting |
| RQ4 | Can Bayesian Networks support clinical inference beyond classification? | Five use cases demonstrated belief updating, partial evidence, risk stratification, and reverse inference |
| RQ5 | Which features drive predictions, and how robust are they? | Glucose was the dominant predictor, with a non-additive interaction between glucose and BMI |
| RQ6 | How does Bayesian Network interpretability compare with SHAP? | The DAG and CPTs provide faithful, model-native explanations; SHAP (post-hoc feature attribution) cannot encode causal direction |

## Dataset

The Pima Indians Diabetes Dataset contains 768 female patients of Pima Indian heritage, each described by eight clinical features:

- Pregnancies
- Glucose
- Blood pressure
- Skin thickness
- Insulin
- Body mass index
- Diabetes pedigree function
- Age

The target is a binary diagnosis of diabetes. The dataset contains 500 non-diabetic patients (65.1%) and 268 diabetic patients (34.9%), a moderate class imbalance of approximately 1.9:1.

### Preprocessing

- Treated biologically impossible zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI as missing values.
- Replaced those zeros with the feature mean, following the approach used in the referenced literature.
- Discretised continuous features into three or four clinically plausible categories so they could be represented in conditional probability tables.
- Split the prepared data 80/20 with stratification: 614 training samples and 154 held-out test samples.
- Used the training set for structure learning, model fitting, and cross-validation; the test set was reserved for final evaluation.

The dataset is included as [`diabetes.csv`](diabetes.csv). It is small, dated, and restricted to Pima Indian women, so the findings should not be generalised to other populations without external validation.

## Models compared

| Model | Role |
| --- | --- |
| Expert Bayesian Network | 12 clinically justified directed edges, built from diabetes literature and pathophysiology |
| Learned Bayesian Network | Structure learned from training data using hill climbing with K2 scoring |
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Bagged tree ensemble |
| XGBoost | Gradient-boosted tree ensemble |
| Neural Network | Multi-layer perceptron with two hidden ReLU layers |

Classical models were trained on the training set and assessed with 10-fold cross-validation. Final performance was measured once on the 20% holdout set. Bayesian Network probabilities were clipped to the range 0.05 to 0.95 to reduce the effect of sparse conditional probability table cells.

## Results

### Classification performance

Results on the 154-patient holdout set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian Network | 0.7273 | 0.6200 | 0.5741 | 0.5962 | 0.7202 |
| Logistic Regression | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8207 |
| Random Forest | 0.7403 | 0.6250 | 0.6481 | 0.6364 | 0.7544 |
| XGBoost | 0.7143 | 0.5926 | 0.5926 | 0.5926 | 0.7526 |
| Neural Network | **0.7727** | **0.7021** | 0.6111 | **0.6535** | **0.8431** |

The Neural Network achieved the strongest discrimination, but the Bayesian Network remained within 4.55 percentage points of accuracy and 12.29 percentage points of ROC-AUC while offering a transparent graph structure, interpretable conditional probability tables, and inference capabilities that the other models do not provide.

Random Forest had the highest recall at 0.6481, which matters if the priority is identifying as many high-risk patients as possible.

### Probability calibration

| Model | Brier score | Log Loss | ECE |
| --- | ---: | ---: | ---: |
| Bayesian Network | 0.2230 | 0.7170 | 0.1960 |
| Logistic Regression | **0.1627** | **0.4823** | **0.0778** |
| Random Forest | 0.2006 | 0.8155 | 0.1556 |
| XGBoost | 0.2171 | 0.6727 | 0.1916 |
| Neural Network | 0.1788 | 0.5340 | 0.0939 |
| Base-rate baseline | 0.2277 | n/a | n/a |

Lower is better for all three metrics. Logistic Regression was the best-calibrated model despite not having the highest accuracy. The Bayesian Network's Brier score was only slightly better than the base-rate baseline, and its ECE of 0.1960 shows that its probabilities still require improvement before clinical use.

This distinction is important: a model can rank patients reasonably well while producing probabilities that are poorly calibrated. Accuracy alone is not enough for a clinical decision-support context.

### Expert DAG versus learned DAG

| Metric | Expert DAG | Learned DAG |
| --- | ---: | ---: |
| Accuracy | **0.7273** | 0.7078 |
| ROC-AUC | **0.7202** | 0.5489 |
| Directed edges | 12 | 16 |
| Structural Hamming Distance vs expert | 0 | 16 |
| Edge recall vs expert | reference | 0.500 |
| Edge precision vs expert | reference | 0.375 |

The learned DAG added edges, reversed some clinically meaningful directions, and collapsed to an ROC-AUC of 0.5489, barely above random chance. The expert DAG's better Bayesian Information Criterion (BIC) score and performance across all evaluation metrics indicate that clinical knowledge acted as a regulariser on this small sample.

### Probabilistic inference

The Bayesian Network was used for five inference scenarios that standard classifiers cannot handle directly:

- **Prior risk:** 34.6% probability of diabetes with no evidence, closely matching the 34.85% training base rate.
- **Incremental belief update:** adding advanced age raised risk to 53.3%; adding high glucose raised it to 84.3%; adding obesity raised it to 90.1%.
- **Partial evidence:** predictions with only age, glucose, and BMI reached 90.1%; adding blood pressure and insulin raised this to 90.6%; all features produced 93.75%.
- **Risk stratification:** representative profiles ranged from 5.0% for a young healthy patient to 95.0% for a very high-risk patient.
- **Reverse inference:** the probability of high glucose given diabetes was 49.53%, compared with 10.50% given no diabetes.

### Sensitivity and robustness

- Glucose had the largest one-at-a-time sensitivity range: 90.00 percentage points.
- Diabetes Pedigree Function ranked second at 82.55 points, followed by age at 32.85 and BMI at 30.79.
- Glucose and BMI showed a non-additive interaction; the joint high-risk combination exceeded the additive approximation.
- Reducing glucose by one category for a maximum-risk patient lowered predicted risk from 96.8% to 77.4%, a drop of 19.4 percentage points.

## Key findings

1. **Bayesian Networks are competitively accurate.** The 4.55 percentage point accuracy gap to the best Neural Network is modest, especially given the Bayesian Network's interpretability and inference capabilities.
2. **Accuracy and calibration are different questions.** Logistic Regression was better calibrated than the more accurate Neural Network, and the Bayesian Network needed probability clipping to handle sparse CPT cells.
3. **Expert knowledge matters on small data.** The expert DAG outperformed the learned DAG on all metrics. Unconstrained structure learning overfit 614 training samples and failed to generalise.
4. **Probabilistic inference adds clinical utility.** Missing-data handling, belief updating, stratification, and reverse inference are useful capabilities that a standard classifier does not expose.
5. **Glucose is the dominant predictor.** Sensitivity analysis and the glucose-BMI interaction support glucose as the primary risk driver, while showing that linear additive assumptions would miss an important interaction.

## Repository contents

```text
.
├── Diabetes_Prediction_Bayesian_vs_ML.ipynb   # Full analysis and experiments
├── diabetes.csv                               # Pima Indians Diabetes Dataset
└── README.md
```

Running the notebook from the repository root generates its figures and metric files in the working directory, including confusion matrices, ROC curves, reliability diagrams, DAG comparisons, sensitivity plots, and CSV summaries.

## Getting started

### Requirements

- Python 3.10
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/iamkeerat/Diabetes-Prediction-Bayesian-vs-ML.git
cd Diabetes-Prediction-Bayesian-vs-ML

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn xgboost bnlearn pgmpy networkx jupyter
```

### Run the analysis

```bash
jupyter notebook Diabetes_Prediction_Bayesian_vs_ML.ipynb
```

Run the cells sequentially from the repository root so that `diabetes.csv` resolves correctly. The notebook covers exploratory data analysis, preprocessing, model training, calibration analysis, Bayesian structure learning, inference, and sensitivity analysis.

For Google Colab, upload both the notebook and `diabetes.csv`, then run the notebook from the uploaded files.

## Reproducibility notes

- The 80/20 stratified split and 10-fold cross-validation setup are implemented in the notebook.
- Fixed random seeds of 42 are used for the split, classical models, and cross-validation.
- Results in this README are the held-out test results reported in the dissertation.
- Figures and CSV outputs are regenerated when the notebook is run.
- The notebook is the source of truth; this README summarises the dissertation.
- Small numerical differences may occur across library versions or environments.

## Limitations and responsible use

- The dataset has only 768 patients, dates from 1990, and contains only female patients of Pima Indian heritage.
- Mean imputation and discretisation discard information and may affect calibration and inference.
- The notebook applies imputation and discretisation to the full dataset before the train-test split, which can introduce a small amount of preprocessing leakage. A stricter workflow would fit imputation values and bin boundaries on the training set only.
- The small training sample limits the reliability of data-learned structures; larger datasets may change the expert-versus-learned comparison.
- Interpretability was evaluated through the model structure and a theoretical comparison with SHAP, not through a clinician user study.
- This is an academic research project, not a medical device. It must not be used for clinical diagnosis or treatment decisions without external validation, regulatory review, and clinician involvement.

## Citation

```bibtex
@mastersthesis{singh2026bayesian,
  author  = {Keerat Anmol Singh},
  title   = {Bayesian Networks for Diabetes Prediction: A Comparative Analysis with Classical Machine Learning Models},
  school  = {University of Huddersfield},
  year    = {2026},
  type    = {MSc dissertation},
  note    = {Supervisor: Dr Zahid Hussain}
}
```

## Author

**Keerat Anmol Singh**  
MSc Artificial Intelligence, University of Huddersfield  
[GitHub](https://github.com/iamkeerat)

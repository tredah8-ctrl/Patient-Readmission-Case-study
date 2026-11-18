# Patient-Readmission-Case-study
# 30-Day Hospital Patient Readmission Prediction Using Machine Learning

# 1. Project Overview

This project focuses on analysing hospital patient data and developing machine-learning models to predict whether a patient is likely to be readmitted to hospital within 30 days after discharge.

Hospital readmissions are an important healthcare concern because they can increase healthcare costs, place additional pressure on hospital resources and affect patient outcomes.

The purpose of this project is to explore patient information, identify possible patterns associated with readmission, build predictive models and evaluate how accurately machine learning can predict readmission.

The project uses Python and machine-learning techniques to investigate the relationship between patient characteristics and hospital readmission outcomes.

---

#2. Project Objectives

The main objectives are to:

1. Understand the structure and quality of the patient dataset.
2. Clean and prepare the data for analysis.
3. Perform Exploratory Data Analysis (EDA) to identify patterns and trends.
4. Investigate relationships between patient characteristics and readmission.
5. Select relevant features for machine-learning modelling.
6. Build and compare different machine-learning models.
7. Evaluate model performance using classification metrics.
8. Identify limitations and provide recommendations based on the findings.

---

# 3. Dataset Description

The project uses the dataset:

`Expanded_Patient_Readmission_Data_1.csv`

The dataset contains 3,000 patient records and 10 columns.

# Dataset variables

| Column | Description |
|---|---|
| Patient ID | Unique patient identifier |
| Age | Patient's age |
| Gender | Patient's gender |
| Admission Type | Elective or emergency admission |
| Length of Stay | Number of days spent in hospital |
| Number of Diagnoses | Number of diagnoses recorded |
| Blood Pressure | Patient's blood pressure measurement |
| Blood Sugar Levels | Patient's blood sugar measurement |
| Previous Admissions | Number of previous hospital admissions |
| Readmission | Whether the patient was readmitted within 30 days |

# Target variable

The target variable is `Readmission`.

- **Yes:** The patient was readmitted within 30 days.
- **No:** The patient was not readmitted within 30 days.

The target variable was converted into numerical values for machine learning:

- No = 0
- Yes = 1

# Target distribution

The dataset contains:

- Total patients: 3,000
- Patients not readmitted: 2,134 (71.13%)
- Patients readmitted: 866 (28.87%)

This shows that the dataset is imbalanced because there are more patients who were not readmitted than patients who were readmitted.

---

# 4. Tools and Technologies

The following tools and libraries were used or planned for the project:

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation, where used |
| Scikit-learn | Machine-learning models and evaluation |
| Jupyter Notebook / Databricks | Development environment |
| Microsoft Word | Report preparation |
| PowerPoint / Power BI | Presentation and dashboard development, where required |

---

# 5. Project Methodology

The project followed a structured data science workflow.

# Task 1: Data Understanding and Preprocessing

The first stage involved loading the dataset and understanding its structure.

Activities included:

- Loading the CSV dataset using Pandas.
- Inspecting the first and last records.
- Checking the number of rows and columns.
- Reviewing column names and data types.
- Checking for missing values.
- Checking for duplicate records.
- Reviewing unique values in categorical columns.
- Examining numerical variables and their ranges.

The dataset contained 3,000 records and 10 columns. The initial checks indicated that there were no missing values or duplicate rows.

The `Patient ID` column was treated as an identifier and excluded from model training because it does not represent a meaningful patient characteristic.

Categorical variables such as `Gender` and `Admission Type` were converted into numerical columns using one-hot encoding with `pd.get_dummies()`.

The data was divided into:

- Training data: 2,400 records (80%)
- Testing data: 600 records (20%)

A stratified split was used to maintain approximately the same readmission class distribution in both datasets.

# Task 2: Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the characteristics of the dataset and investigate possible patterns in hospital readmission.

The analysis included:

- Distribution of readmission outcomes.
- Patient age distribution.
- Gender distribution.
- Admission type distribution.
- Length of hospital stay.
- Number of diagnoses.
- Blood pressure distribution.
- Blood sugar level distribution.
- Previous hospital admissions.
- Comparisons between readmitted and non-readmitted patients.

Visualisations were used to make the findings easier to understand.

The EDA showed that readmission was not evenly distributed across the target classes. Approximately 28.87% of patients were readmitted, while 71.13% were not.

The average age of patients who were readmitted was approximately 54.42 years, compared with 53.06 years for patients who were not readmitted.

The differences between these group averages were relatively small.

The analysis also investigated readmission rates across gender, admission type and previous admissions.

# Task 3: Feature Selection and Analysis

Feature selection was performed to investigate which variables might be useful for predicting readmission.

The analysis included:

- Chi-square tests for categorical variables.
- Cramér's V to measure categorical associations.
- Mutual information to investigate relationships with the target.
- Permutation feature importance.
- Correlation analysis between numerical variables.

# Categorical feature analysis

The following results were obtained:

| Feature | Chi-square | P-value | Cramér's V |
|---|---:|---:|---:|
| Gender | 0.602149 | 0.437759 | 0.014167 |
| Admission Type | 0.347973 | 0.555263 | 0.010770 |

Both variables showed very weak associations with the readmission outcome in these tests.

# Mutual information

The highest mutual information scores were:

| Feature | Mutual Information |
|---|---:|
| Blood Sugar Levels | 0.006999 |
| Previous Admissions | 0.004344 |
| Length of Stay | 0.002378 |
| Age | 0.001279 |
| Gender | 0.000111 |
| Admission Type | 0.000066 |
| Number of Diagnoses | 0.000000 |
| Blood Pressure | 0.000000 |

The scores were generally low, suggesting that the available variables contained limited predictive information about readmission.

These results do not prove that the variables have no clinical importance. They indicate that their predictive relationships were weak in this dataset and analysis.

# Task 4: Machine-Learning Model Development

Several classification models were developed to predict whether a patient would be readmitted within 30 days.

The models considered were:

1. Dummy Classifier (baseline model)
2. Logistic Regression
3. Decision Tree
4. Random Forest

The baseline model was used as a reference point to compare the performance of the other models.

A preprocessing pipeline was used to handle numerical and categorical variables. Numerical variables were standardised, while categorical variables were one-hot encoded.

Five-fold Stratified Cross-Validation was used to assess model performance on the training data.

ROC-AUC was selected as the main cross-validation metric.

# Cross-validation results

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Dummy Classifier | 0.5000 | 0.0000 |
| Logistic Regression | 0.4941 | 0.0262 |
| Decision Tree | 0.5143 | 0.0168 |
| Random Forest | 0.5130 | 0.0185 |

The Decision Tree achieved the highest mean cross-validation ROC-AUC of approximately 0.5143.

However, all models performed close to 0.50, which indicates weak discrimination between patients who were and were not readmitted.

**Important project status:** Hyperparameter tuning for at least two models remains to be completed and documented before Task 4 can be considered fully complete.

# Task 5: Model Evaluation

The Decision Tree model was provisionally selected based on its cross-validation result and evaluated using the held-out test dataset.

The following evaluation metrics were obtained:

| Metric | Result |
|---|---:|
| Accuracy | 59.67% |
| Precision | 29.09% |
| Recall | 27.75% |
| F1-score | 28.40% |
| ROC-AUC | 0.5017 |

# Interpretation

**Accuracy:** The model correctly classified approximately 59.67% of the test records.

**Precision:** Of the patients predicted to be readmitted, approximately 29.09% were actually readmitted.

**Recall:** The model identified approximately 27.75% of the patients who were actually readmitted.

**F1-score:** The F1-score of 28.40% shows that the balance between precision and recall was weak.

**ROC-AUC:** The ROC-AUC of 0.5017 is very close to 0.50, indicating that the model has almost no ability to distinguish between the two outcome classes.

These results show that the model is not yet reliable enough for practical clinical use.

Further evaluation includes the confusion matrix, ROC curve, precision-recall curve, threshold analysis and investigation of overfitting.

---

# 6. Key Findings

The main findings from the analysis include:

1. The dataset contains more patients who were not readmitted than patients who were readmitted.
2. Approximately 28.87% of patients in the dataset experienced readmission within 30 days.
3. The average ages of readmitted and non-readmitted patients were relatively similar.
4. Gender and admission type showed very weak statistical associations with readmission in the tests performed.
5. Mutual information scores were low across the available features.
6. The Decision Tree achieved the highest mean ROC-AUC among the initial cross-validation results, but its performance was only slightly above the baseline.
7. The final test ROC-AUC was approximately 0.5017, showing that the model's predictive performance was very limited.

---

# 7. Challenges and Limitations

Several limitations were identified during the project.

# Limited predictive performance

The models produced ROC-AUC results close to 0.50, indicating that the available features were insufficient for reliable prediction.

# Class imbalance

The dataset contained more non-readmitted patients than readmitted patients. This can affect how well a model identifies the smaller readmission class.

# Limited patient information

The dataset does not include every factor that could influence hospital readmission.

Additional useful information could include:

- Existing medical conditions and comorbidities.
- Medication history.
- Previous treatment details.
- Discharge planning information.
- Follow-up appointments.
- Socioeconomic factors.
- Patient lifestyle information.

# Overfitting risk

A model may perform well on training data but poorly on unseen data. Training and testing performance must therefore be compared.

# Clinical limitations

The model is intended for educational and analytical purposes. It should not be used as the only basis for clinical decisions or patient treatment.

---

# 8. Recommendations

Based on the findings, the following recommendations are proposed:

1. Collect additional patient information that may improve readmission prediction.
2. Complete hyperparameter tuning and compare the tuned models.
3. Investigate class-imbalance techniques and their effect on recall and precision.
4. Evaluate different classification thresholds using validation data.
5. Pay particular attention to false negatives because these represent patients who were readmitted but predicted as not readmitted.
6. Conduct further testing using a larger and more representative dataset.
7. Ensure that any future healthcare prediction system is reviewed by qualified healthcare professionals.

---

# 9. Project Deliverables

The project deliverables include:

- Jupyter Notebook (`.ipynb`) containing the analysis and modelling workflow.
- Python source code.
- Exploratory Data Analysis report.
- At least six data visualisations.
- Feature selection analysis.
- Machine-learning model comparison.
- Model evaluation results.
- Final project report.
- Project presentation.
- Project plan and Gantt chart.

---

# 10. Project Timeline

The project was organised into a three-week schedule.

| Week | Main Activities |
|---|---|
| Week 1 | Project planning, dataset understanding, data cleaning and EDA |
| Week 2 | Data preprocessing, feature selection and machine-learning model development |
| Week 3 | Model evaluation, interpretation, recommendations, final report and presentation |

---

# 11. Conclusion

This project explored the use of machine learning to predict 30-day hospital patient readmission.

The work included data understanding, preprocessing, exploratory data analysis, feature selection, model development and model evaluation.

Although several classification models were developed, their predictive performance was weak. The Decision Tree achieved the highest initial cross-validation ROC-AUC, but its final test ROC-AUC was approximately 0.5017.

This demonstrates that building a machine-learning model does not necessarily guarantee accurate predictions. Data quality, feature relevance, model selection and evaluation are all important parts of the data science process.

Further work is required to improve predictive performance, investigate additional patient information and complete model tuning.

The project provides practical experience in applying Python, data analysis and machine-learning techniques to a healthcare-related problem.

---

# 12. How to Run the Project

# Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

# Steps

1. Download or obtain the project dataset.
2. Place `Expanded_Patient_Readmission_Data_1.csv` in the appropriate project directory.
3. Open the Jupyter Notebook or Databricks notebook.
4. Run the notebook cells in order, starting with data loading and exploration.
5. Complete preprocessing, feature selection, model training and evaluation.
6. Review the generated charts and model results.

# Example: Load the dataset

```python
import pandas as pd

# Load the patient readmission dataset
df = pd.read_csv("Expanded_Patient_Readmission_Data_1.csv")

# Display the first five rows
df.head()
```

---

# 13. Project Status

The project has completed the initial dataset exploration, EDA, preprocessing, feature analysis, initial model development and initial test evaluation.

Remaining work includes completing hyperparameter tuning for at least two models and finalising the remaining evaluation plots, interpretation and submission documents.

**Project title:** 30-Day Hospital Patient Readmission Prediction Using Machine Learning

**Project type:** Data Science / Machine Learning

**Dataset:** Expanded_Patient_Readmission_Data_1.csv

**Programming language:** Python

# Task 1 Submission Document: ML Classification Project

## Objective
To build a supervised machine learning classification system that predicts a user's mental health risk level using psychological indicators collected from social media and behavioral data.

## Problem Statement
Mental health risk identification can be difficult when psychological signals are spread across multiple behavioral indicators. This project uses machine learning to classify `mental_health_risk` into risk categories such as Low, Medium, and High based on selected numerical indicators.

## Dataset Description
The dataset contains the following columns:

| Column | Description |
|---|---|
| `user_id` | Anonymous user identifier |
| `year` | Year of record |
| `country` | User country |
| `age_group` | User age category |
| `gender` | User gender |
| `platform` | Social media platform |
| `anxiety_score` | Anxiety indicator score |
| `depression_score` | Depression indicator score |
| `stress_level` | Stress indicator score |
| `loneliness_index` | Loneliness indicator score |
| `therapy_access` | Therapy access status |
| `medication_usage` | Medication usage status |
| `self_esteem_score` | Self-esteem indicator score |
| `mental_health_risk` | Target class |

## Methodology
1. Loaded the dataset using Pandas.
2. Performed data understanding using `head()`, `info()`, and summary statistics.
3. Checked and handled missing values.
4. Removed duplicate rows.
5. Encoded the target label using `LabelEncoder`.
6. Selected model features:
   - `anxiety_score`
   - `depression_score`
   - `stress_level`
   - `loneliness_index`
   - `self_esteem_score`
7. Split the data into 80% training and 20% testing using `random_state=42`.
8. Trained Logistic Regression and Random Forest models.
9. Evaluated models using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
10. Performed 5-fold cross-validation.

## Tools and Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## Code Explanation
The notebook and `train_model.py` load the dataset, clean missing values, encode labels, select features, train models, evaluate performance, generate visualizations, and save the best model as `models/model.pkl`. The label encoder is saved as `models/label_encoder.pkl`.

## Screenshots Placeholders
[INSERT SCREENSHOT HERE]

Dataset preview screenshot.

[INSERT SCREENSHOT HERE]

EDA visualizations screenshot.

[INSERT SCREENSHOT HERE]

Confusion matrix screenshot.

## Results and Outputs

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.4166 | 0.3642 | 0.4166 | 0.3595 | 0.4993 |
| Logistic Regression | 0.4489 | 0.2015 | 0.4489 | 0.2782 | 0.4981 |

Random Forest achieved the better weighted F1 score, while Logistic Regression achieved higher raw accuracy. This indicates that class imbalance and weak feature separation affect model behavior.

## Conclusion
The ML classification workflow was completed successfully. The Random Forest model was selected for deployment because it produced the stronger weighted F1 score and provides feature importance for interpretation.

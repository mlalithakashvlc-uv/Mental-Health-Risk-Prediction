# Project Report: Mental Health Risk Prediction Using Machine Learning

## Abstract
This project develops an end-to-end machine learning classification system for predicting mental health risk levels using psychological and behavioral indicators. The workflow includes data understanding, preprocessing, exploratory data analysis, Logistic Regression, Random Forest classification, model evaluation, responsible AI analysis, and deployment through Flask and Streamlit.

## Introduction
Mental health concerns are increasingly visible across digital platforms. Machine learning can help identify patterns in psychological indicators and provide early awareness. This project predicts `mental_health_risk` using structured indicators such as anxiety, depression, stress, loneliness, and self-esteem scores.

## Literature Overview
Machine learning has been widely used for mental health screening, sentiment analysis, behavioral risk prediction, and clinical decision support. Traditional classifiers such as Logistic Regression provide interpretable baselines, while ensemble methods such as Random Forest can capture non-linear patterns. However, mental health prediction requires careful ethical handling because model outputs may be misunderstood as clinical diagnosis.

## Dataset Description
The dataset contains demographic, platform, psychological, and treatment-related variables:

`user_id`, `year`, `country`, `age_group`, `gender`, `platform`, `anxiety_score`, `depression_score`, `stress_level`, `loneliness_index`, `therapy_access`, `medication_usage`, `self_esteem_score`, `mental_health_risk`.

The target variable is `mental_health_risk`.

## Data Preprocessing
Preprocessing steps included:

- Checked missing values.
- Filled numeric missing values with median.
- Filled categorical missing values with mode.
- Removed duplicate records.
- Encoded target labels using `LabelEncoder`.
- Selected five numerical features for deployment.

## Exploratory Data Analysis
EDA visualizations generated:

- Histogram of `anxiety_score`
- Histogram of `depression_score`
- Boxplot of `stress_level`
- Countplot of `mental_health_risk`
- Scatterplot of `anxiety_score` vs `depression_score`
- Correlation heatmap
- Gender distribution bar chart
- Platform distribution bar chart

[INSERT SCREENSHOT HERE]

## Machine Learning Models
Two classification models were trained:

1. Logistic Regression
2. Random Forest Classifier

The dataset was split into 80% training and 20% testing with `random_state=42`.

## Logistic Regression Results
Logistic Regression provided a baseline model. It achieved higher raw accuracy but lower weighted F1 score, indicating that it mainly favored the majority class.

| Metric | Score |
|---|---:|
| Accuracy | 0.4489 |
| Precision | 0.2015 |
| Recall | 0.4489 |
| F1 Score | 0.2782 |
| ROC-AUC | 0.4981 |

## Random Forest Results
Random Forest produced a better weighted F1 score and enabled feature importance interpretation.

| Metric | Score |
|---|---:|
| Accuracy | 0.4166 |
| Precision | 0.3642 |
| Recall | 0.4166 |
| F1 Score | 0.3595 |
| ROC-AUC | 0.4993 |

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.4166 | 0.3642 | 0.4166 | 0.3595 | 0.4993 |
| Logistic Regression | 0.4489 | 0.2015 | 0.4489 | 0.2782 | 0.4981 |

Random Forest was selected for deployment due to its better weighted F1 score and interpretability through feature importance.

## Responsible AI Analysis
The model should not be used as a diagnostic tool. Bias may exist across gender, country, age group, and platform usage. The dataset may not be clinically validated or demographically balanced. Predictions must be interpreted as educational support only.

## Deployment Overview
The trained model was saved using Joblib:

- `models/model.pkl`
- `models/label_encoder.pkl`

Deployment interfaces:

- Flask web application
- Streamlit web application
- Docker container preparation

[INSERT SCREENSHOT HERE]

## Conclusion
The project successfully demonstrates an end-to-end machine learning pipeline, from data exploration to model deployment. Random Forest was selected as the final model based on weighted F1 score and interpretability.

## Future Scope
- Improve dataset quality with clinically validated labels.
- Add fairness metrics across demographic groups.
- Use NLP features from social media text.
- Improve class imbalance handling.
- Add explainability tools such as SHAP or LIME.
- Deploy through a secure production environment.

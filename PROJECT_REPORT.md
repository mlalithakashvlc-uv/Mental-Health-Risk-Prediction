# Project Report: Mental Health Risk Prediction

## Objective
Build a supervised classification system to predict a person's `mental_health_risk` using psychological and behavioral indicators.

## Data Understanding
The dataset includes user metadata, social media platform information, psychological scores, treatment access indicators, and the target risk class. The notebook and `train_model.py` display the first records, dataset info, summary statistics, and column explanations.

## Preprocessing
Missing numerical values are filled with medians. Missing categorical values are filled with modes. Duplicate rows are removed. The target variable is encoded with `LabelEncoder`. Categorical variables are encoded for broader analysis, while the final model uses selected numeric psychological indicators.

## EDA
The project generates histograms for anxiety and depression, a boxplot for stress, countplots for risk labels, gender, and platform, a scatterplot for anxiety vs depression, and a correlation heatmap.

## Modeling
Two classifiers are trained with an 80/20 split and `random_state=42`:
- Logistic Regression
- Random Forest Classifier

Evaluation metrics include accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and 5-fold cross-validation accuracy.

## Feature Importance
Random Forest feature importance is plotted to show which psychological indicators contribute most to predictions. High anxiety, depression, stress, loneliness, and low self-esteem may all influence risk, depending on learned dataset patterns.

## Responsible AI and Ethics
Mental health data is sensitive. Bias may appear across gender, country, age group, platform usage, or therapy access. The dataset may not represent all populations or real clinical conditions. Predictions can support awareness and triage, but they should never replace licensed professional diagnosis or crisis intervention.

## Conclusion
The best model is selected by weighted F1 score and saved for deployment. The project demonstrates a full ML workflow from data understanding through Flask deployment.

## Deployment
The trained model and label encoder are saved as `models/model.pkl` and `models/label_encoder.pkl`. The Flask app accepts five numeric inputs and returns the predicted risk class.

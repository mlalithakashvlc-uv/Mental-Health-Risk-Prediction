# Mental Health Risk Prediction Using Social Media and Psychological Indicators

Supervised ML classification project that predicts `mental_health_risk` from psychological indicators.

## Dataset
Expected columns:
`user_id`, `year`, `country`, `age_group`, `gender`, `platform`, `anxiety_score`, `depression_score`, `stress_level`, `loneliness_index`, `therapy_access`, `medication_usage`, `self_esteem_score`, `mental_health_risk`.

The training script auto-loads the existing dataset from:
`../projects/mental_health_trends/dataset/mental_health_trends.csv`

You can also place a CSV at:
`data/mental_health_trends.csv`

## Setup
```bash
pip install -r requirements.txt
```

## Train and Evaluate
```bash
python train_model.py
```

This creates:
- `models/model.pkl`
- `models/label_encoder.pkl`
- EDA and evaluation charts in `outputs/figures/`
- `outputs/model_comparison.csv`

## Run Streamlit App
```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501/`.

## Deploy on Streamlit Community Cloud
1. Push this project folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the repository and set the main file path to `streamlit_app.py`.
4. Keep `requirements.txt` in the repository root.
5. Make sure `models/model.pkl` and `models/label_encoder.pkl` are included, or run training before deployment in your own hosting setup.

## Run Flask App
```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## Features Used for Modeling
- `anxiety_score`
- `depression_score`
- `stress_level`
- `loneliness_index`
- `self_esteem_score`

## Responsible AI Note
This system is for learning and decision support only. It must not replace professional mental health screening, diagnosis, or emergency care.
"# Mental-Health-Risk-Prediction" 
"# Mental-Health-Risk-Prediction" 

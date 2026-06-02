# Task 3 Submission Document: Model Deployment - API & Containerization

## Objective
To deploy the trained machine learning model through a web application and prepare the project for containerized deployment.

## Problem Statement
A trained ML model is useful only when users can interact with it easily. This task focuses on creating a deployment interface where users enter psychological indicator scores and receive a predicted mental health risk level.

## Dataset Description
The deployed model uses five selected input features:

| Input Parameter | Type | Description |
|---|---|---|
| `anxiety_score` | Float | Anxiety score from 0 to 100 |
| `depression_score` | Float | Depression score from 0 to 100 |
| `stress_level` | Float | Stress score from 0 to 100 |
| `loneliness_index` | Float | Loneliness index from 0 to 100 |
| `self_esteem_score` | Float | Self-esteem score from 0 to 100 |

The output is the predicted `mental_health_risk` class.

## Methodology
1. Trained and saved the best ML model using Joblib.
2. Saved the target label encoder.
3. Built a Flask web app for browser-based prediction.
4. Built a Streamlit app for simple cloud deployment.
5. Added dependency list in `requirements.txt`.
6. Prepared Docker containerization files.

## Tools and Technologies Used
- Python
- Flask
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Docker

## Flask Architecture
The Flask app follows a simple MVC-style flow:

1. User opens the home page.
2. User submits five numeric input values.
3. Flask route receives form data through `POST`.
4. Saved model and label encoder are loaded.
5. Input values are converted into a Pandas DataFrame.
6. Model predicts the encoded class.
7. Label encoder converts the class back to a readable risk label.
8. Prediction is displayed on the web page.

## API Endpoint

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Displays prediction form |
| `POST` | `/` | Accepts input scores and returns prediction |

## Streamlit Deployment
The Streamlit app provides a clean deployment interface using sliders for all five input variables. It loads the same `model.pkl` and `label_encoder.pkl` files.

Run command:

```bash
python -m streamlit run streamlit_app.py
```

## Prediction Output
The application returns one predicted class:

```text
Predicted Mental Health Risk: Medium
```

## Containerization Explanation
Docker packages the application, source code, model files, and dependencies into a reproducible container. This avoids environment mismatch issues across machines.

Typical Docker workflow:

```bash
docker build -t mental-health-risk-app .
docker run -p 8501:8501 mental-health-risk-app
```

## Screenshots Placeholders
[INSERT SCREENSHOT HERE]

Flask application home page.

[INSERT SCREENSHOT HERE]

Streamlit prediction interface.

[INSERT SCREENSHOT HERE]

Docker build/run terminal output.

## Results and Outputs
- Flask app created successfully.
- Streamlit app created successfully.
- Saved model files loaded correctly.
- Local prediction route validated.
- Streamlit app tested locally on port `8501`.

## Conclusion
The model was successfully prepared for deployment through Flask, Streamlit, and Docker-based containerization.

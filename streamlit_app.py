"""Streamlit app for mental health risk prediction.

Run locally:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"
ENCODER_PATH = PROJECT_DIR / "models" / "label_encoder.pkl"
FEATURES = [
    "anxiety_score",
    "depression_score",
    "stress_level",
    "loneliness_index",
    "self_esteem_score",
]


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        st.error("Model files not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Mental Health Risk Prediction",
        page_icon="🧠",
        layout="centered",
    )

    st.title("Mental Health Risk Prediction")
    st.caption("Educational ML classifier. Not a clinical diagnosis.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            anxiety_score = st.slider("Anxiety Score", 0.0, 100.0, 60.0, 0.1)
            stress_level = st.slider("Stress Level", 0.0, 100.0, 55.0, 0.1)
            self_esteem_score = st.slider("Self Esteem Score", 0.0, 100.0, 50.0, 0.1)
        with col2:
            depression_score = st.slider("Depression Score", 0.0, 100.0, 60.0, 0.1)
            loneliness_index = st.slider("Loneliness Index", 0.0, 100.0, 55.0, 0.1)

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        values = {
            "anxiety_score": anxiety_score,
            "depression_score": depression_score,
            "stress_level": stress_level,
            "loneliness_index": loneliness_index,
            "self_esteem_score": self_esteem_score,
        }
        model, label_encoder = load_artifacts()
        input_df = pd.DataFrame([values], columns=FEATURES)
        encoded_prediction = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([encoded_prediction])[0]

        st.success(f"Predicted Mental Health Risk: {prediction}")

    with st.expander("Responsible AI note"):
        st.write(
            "This app is for education and decision support only. Mental health "
            "predictions can be biased or incomplete and should not replace "
            "professional diagnosis, therapy, crisis care, or clinical judgment."
        )


if __name__ == "__main__":
    main()

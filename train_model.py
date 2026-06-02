"""Train classification models for mental health risk prediction.

Run:
    python train_model.py

The script loads the CSV, performs data understanding, preprocessing, EDA,
model training, evaluation, cross validation, feature importance, and saves
model artifacts for Flask deployment.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    PROJECT_DIR / "data" / "mental_health_trends.csv",
    PROJECT_DIR.parent / "projects" / "mental_health_trends" / "dataset" / "mental_health_trends.csv",
]
MODEL_DIR = PROJECT_DIR / "models"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "anxiety_score",
    "depression_score",
    "stress_level",
    "loneliness_index",
    "self_esteem_score",
]
TARGET = "mental_health_risk"

COLUMN_EXPLANATIONS = {
    "user_id": "Unique anonymous user identifier.",
    "year": "Year when the record was collected.",
    "country": "User country.",
    "age_group": "Age segment such as Young Adult, Adult, or Senior.",
    "gender": "Self-reported gender category.",
    "platform": "Social media platform used by the person.",
    "anxiety_score": "Numerical anxiety indicator; higher means more anxiety symptoms.",
    "depression_score": "Numerical depression indicator; higher means more depression symptoms.",
    "stress_level": "Numerical stress indicator.",
    "loneliness_index": "Numerical loneliness/social isolation indicator.",
    "therapy_access": "Whether the person has access to therapy support.",
    "medication_usage": "Whether the person reports medication usage.",
    "self_esteem_score": "Numerical self-esteem indicator; lower values may indicate higher risk.",
    "mental_health_risk": "Target class: predicted mental health risk level.",
}


def find_dataset() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Place mental_health_trends.csv in data/ or keep the original projects/mental_health_trends/dataset path."
    )


def load_data() -> pd.DataFrame:
    path = find_dataset()
    print(f"Loading dataset: {path}")
    return pd.read_csv(path)


def data_understanding(df: pd.DataFrame) -> None:
    print("\nFirst five records:")
    print(df.head())
    print("\nDataset information:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe(include="all"))
    print("\nColumn explanations:")
    for col, meaning in COLUMN_EXPLANATIONS.items():
        print(f"- {col}: {meaning}")


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    before = len(df)
    df = df.drop_duplicates()
    print(f"\nRemoved {before - len(df)} duplicate rows.")

    label_encoder = LabelEncoder()
    df[TARGET] = label_encoder.fit_transform(df[TARGET])

    # Encode non-target categorical columns for EDA or extended modeling.
    categorical_cols = [c for c in df.select_dtypes(exclude="number").columns if c != TARGET]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("\nMissing values after cleaning:")
    print(df.isna().sum().sum())
    print(f"Target labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    return df, label_encoder


def save_eda(raw_df: pd.DataFrame) -> None:
    plots = [
        ("anxiety_hist.png", lambda: sns.histplot(raw_df["anxiety_score"], kde=True, color="#2563eb"), "Anxiety Score Distribution"),
        ("depression_hist.png", lambda: sns.histplot(raw_df["depression_score"], kde=True, color="#16a34a"), "Depression Score Distribution"),
        ("stress_boxplot.png", lambda: sns.boxplot(y=raw_df["stress_level"], color="#f59e0b"), "Stress Level Boxplot"),
        ("risk_countplot.png", lambda: sns.countplot(data=raw_df, x=TARGET, palette="Set2"), "Mental Health Risk Count"),
        ("anxiety_vs_depression.png", lambda: sns.scatterplot(data=raw_df, x="anxiety_score", y="depression_score", hue=TARGET), "Anxiety vs Depression"),
        ("gender_distribution.png", lambda: sns.countplot(data=raw_df, x="gender", palette="Set3"), "Gender Distribution"),
        ("platform_distribution.png", lambda: sns.countplot(data=raw_df, x="platform", palette="Set3"), "Platform Distribution"),
    ]
    for filename, plot_func, title in plots:
        plt.figure(figsize=(8, 5))
        plot_func()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=140)
        plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(raw_df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=140)
    plt.close()


def evaluate_model(name: str, model, X_test, y_test, label_encoder: LabelEncoder) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
    }
    print(f"\n{name} classification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=140)
    plt.close()
    return metrics


def main() -> None:
    raw_df = load_data()
    data_understanding(raw_df)
    save_eda(raw_df)

    clean_df, label_encoder = preprocess(raw_df)
    X = clean_df[FEATURES]
    y = clean_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=40, random_state=42, class_weight="balanced"),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test, label_encoder))
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"{name} 5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    comparison = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    comparison.to_csv(PROJECT_DIR / "outputs" / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(comparison)

    best_model = models[comparison.iloc[0]["Model"]]
    rf_model = models["Random Forest"]
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": rf_model.feature_importances_}).sort_values("Importance")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, x="Importance", y="Feature", color="#2563eb")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=140)
    plt.close()

    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print(f"\nSaved best model to {MODEL_DIR / 'model.pkl'}")
    print(f"Saved label encoder to {MODEL_DIR / 'label_encoder.pkl'}")
    print("Most influential features:")
    print(importance.sort_values("Importance", ascending=False))


if __name__ == "__main__":
    main()

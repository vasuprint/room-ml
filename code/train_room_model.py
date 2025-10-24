"""Train the room classification model and persist the fitted pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Resolve key project paths relative to this file.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "clean" / "cleaned_all_rooms.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "room_classifier.joblib"

FEATURE_COLUMNS = ["department", "duration_hours", "event_period", "seats"]
TARGET_COLUMN = "room"
MIN_EXAMPLES_PER_CLASS = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the cleaned booking dataset and filter rare classes."""
    df = pd.read_excel(DATA_PATH)
    vc = df[TARGET_COLUMN].value_counts()
    valid_rooms = vc[vc >= MIN_EXAMPLES_PER_CLASS].index
    return df[df[TARGET_COLUMN].isin(valid_rooms)].copy()


def build_pipeline() -> Pipeline:
    """Create the preprocessing and model pipeline."""
    numeric_features = ["duration_hours", "seats"]
    categorical_features = ["department", "event_period"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        min_samples_leaf=12,
        ccp_alpha=0.005,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Fit the pipeline and print a short evaluation report."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))

    print("Train accuracy:", round(train_acc, 3))
    print("Test accuracy:", round(test_acc, 3))
    print("\nTest classification report:\n")
    print(classification_report(y_test, pipeline.predict(X_test)))

    return pipeline


def save_model(pipeline: Pipeline) -> None:
    """Persist the fitted pipeline to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model pipeline to {MODEL_PATH.relative_to(BASE_DIR.parent)}")


if __name__ == "__main__":
    data = load_data()
    trained_pipeline = train_and_evaluate(data)
    save_model(trained_pipeline)

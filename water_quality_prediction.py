"""
Water Quality and Potability Prediction
Model training, evaluation, model selection and saving.

Run from the terminal (with (venv) active):
    python water_quality_prediction.py

Outputs:
    figures/confusion_matrix.png
    figures/model_comparison.png
    model_results.csv
    best_model.joblib
"""

import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    FEATURES,
    RANDOM_STATE,
    clean_data,
    load_raw_data,
    make_preprocessor,
    split_data,
)

# ----- Settings -----
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / "best_model.joblib"
RESULTS_PATH = BASE_DIR / "model_results.csv"

CV_FOLDS = 5
SCORING = "f1"                      # tuning metric: F1 on the Potable class
CLASS_NAMES = ["Not Potable (0)", "Potable (1)"]
METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision (Potable)",
    "Recall (Potable)",
    "F1 (Potable)",
    "ROC-AUC",
]


def build_pipeline(classifier):
    """Preprocessing (median imputation + scaling) followed by the classifier."""
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", classifier),
    ])


def get_model_specs():
    """The three algorithms and the hyperparameter grid each one tries."""
    return {
        "Logistic Regression": {
            "pipeline": build_pipeline(
                LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=RANDOM_STATE)
            ),
            "grid": {"model__C": [0.01, 0.1, 1, 10]},
        },
        "Decision Tree": {
            "pipeline": build_pipeline(
                DecisionTreeClassifier(class_weight="balanced",
                                       random_state=RANDOM_STATE)
            ),
            "grid": {
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_leaf": [1, 10, 20],
            },
        },
        "Random Forest": {
            "pipeline": build_pipeline(
                RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)
            ),
            "grid": {
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5],
            },
        },
    }


def get_baselines():
    """Trivial models used as reference points. They ignore the input features."""
    return {
        "Baseline: always Not Potable": build_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "Baseline: always Potable": build_pipeline(
            DummyClassifier(strategy="constant", constant=1)
        ),
    }


def tune_models(X_train, y_train):
    """Tune each model with 5-fold stratified cross-validation on training data only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    searches = {}
    for name, spec in get_model_specs().items():
        print(f"\nTuning {name} ...")
        start = time.time()
        search = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["grid"],
            scoring=SCORING,
            cv=cv,
            refit=True,
        )
        search.fit(X_train, y_train)
        print(f"  Finished in {time.time() - start:.1f} seconds")
        print(f"  Best parameters : {search.best_params_}")
        print(f"  Best CV F1      : {search.best_score_:.4f}")
        searches[name] = search
    return searches


def evaluate(model, X, y):
    """Compute the metrics for one model on one dataset."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "Precision (Potable)": precision_score(y, pred, zero_division=0),
        "Recall (Potable)": recall_score(y, pred, zero_division=0),
        "F1 (Potable)": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def plot_confusion_matrices(fitted, names, X_test, y_test):
    """One confusion matrix per model, on the test set."""
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    for ax, name in zip(axes, names):
        pred = fitted[name].predict(X_test)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred,
            display_labels=["Not Potable", "Potable"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
    fig.suptitle("Confusion Matrices on the Test Set", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/confusion_matrix.png")


def plot_model_comparison(results):
    """Grouped bar chart comparing all models and baselines on the test set."""
    test = results[results["Split"] == "Test"].set_index("Model")[METRICS]
    ax = test.plot.bar(figsize=(12, 6), rot=15)
    ax.set_title("Model Comparison on the Test Set")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/model_comparison.png")


def main():
    # ----- Load, clean and split (same rules as preprocessing.py) -----
    raw = load_raw_data()
    df, _ = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # ----- Phase 4: tune the three models on training data -----
    print("\n" + "=" * 60)
    print("MODEL TUNING (5-fold stratified CV on training data)")
    print("=" * 60)
    searches = tune_models(X_train, y_train)

    # ----- Fit the baselines on training data -----
    baselines = get_baselines()
    for model in baselines.values():
        model.fit(X_train, y_train)

    fitted = {name: s.best_estimator_ for name, s in searches.items()}
    fitted.update(baselines)

    # ----- Phase 5: evaluate on training and testing data -----
    rows = []
    for name, model in fitted.items():
        for split_name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            scores = evaluate(model, X, y)
            scores["Model"] = name
            scores["Split"] = split_name
            rows.append(scores)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    for split_name in ["Train", "Test"]:
        print("\n" + "=" * 60)
        print(f"{split_name.upper()} SET RESULTS")
        print("=" * 60)
        table = results[results["Split"] == split_name].set_index("Model")[METRICS]
        print(table.round(4).to_string())

    print("\n" + "=" * 60)
    print("TEST SET CLASSIFICATION REPORTS")
    print("=" * 60)
    for name in searches:
        pred = fitted[name].predict(X_test)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, pred, target_names=CLASS_NAMES,
                                    digits=3, zero_division=0))

    # ----- Charts -----
    plot_confusion_matrices(fitted, list(searches), X_test, y_test)
    plot_model_comparison(results)

    # ----- Model selection (uses CV F1 from training data, NOT the test set) -----
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    for name in searches:
        print(f"{name:<22} CV F1 = {searches[name].best_score_:.4f}")
    print(f"\nSelected model: {best_name}")
    print("Reason: highest cross-validated F1 on the Potable class, measured on training data only.")

    # ----- Save the selected model with its metadata -----
    joblib.dump({
        "model_name": best_name,
        "best_params": searches[best_name].best_params_,
        "features": FEATURES,
        "pipeline": fitted[best_name],
    }, MODEL_PATH)
    print(f"Saved: best_model.joblib")

    # ----- Check that the saved file loads and predicts -----
    bundle = joblib.load(MODEL_PATH)
    sample = X_test.iloc[[0]]
    prob = bundle["pipeline"].predict_proba(sample)[0, 1]
    print(f"Reload check: {bundle['model_name']} gives P(Potable) = {prob:.3f} for test row 0")
    print("\nPhase 4-6 complete.")


if __name__ == "__main__":
    main()
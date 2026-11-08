"""Reusable code for the credit-risk project.

The notebook imports from here so the analysis stays readable and the
modelling logic is written (and tested) only once.

Design choices
--------------
* Scaling and one-hot encoding live inside a scikit-learn ``Pipeline`` so
  they are fitted on the training data only. Scaling the full dataset before
  splitting would leak test-set information into training ("data leakage").
* ``class_weight="balanced"`` is used because only ~30% of loans default;
  without it, models drift towards predicting "no default" for everyone.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "german_credit_raw.csv"

# Column names in the raw file -> names used in this project
RENAME = {
    "status": "checking_account",
    "duration": "duration_months",
    "amount": "loan_amount",
    "employment_duration": "employment_length",
}


def load_data(path=DATA_PATH):
    """Load the German Credit data and create the ``Default`` target.

    In the source file ``credit_risk`` is 1 = good, 0 = bad, so we flip it to
    match the project definition: Default 1 = default, 0 = no default.
    """
    df = pd.read_csv(path).rename(columns=RENAME)
    df["Default"] = 1 - df["credit_risk"]
    return df.drop(columns="credit_risk")


def split_columns(X):
    """Return (numeric_cols, categorical_cols) for a feature DataFrame."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocessor(X, scale=True):
    """One-hot encode categoricals; optionally standardise numerics.

    Tree-based models do not need scaling, but it is harmless and keeps the
    three models on an identical input representation.
    """
    num, cat = split_columns(X)
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_step, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
    ])


def build_models(X):
    """Return {name: (pipeline, param_grid)} for the three models."""
    def pipe(clf):
        return Pipeline([("prep", make_preprocessor(X)), ("clf", clf)])

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(class_weight="balanced", max_iter=2000,
                                    random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(class_weight="balanced",
                                        random_state=RANDOM_STATE)),
            {"clf__max_depth": [3, 4, 5, 7],
             "clf__min_samples_leaf": [5, 10, 20]},
        ),
        "Random Forest": (
            pipe(RandomForestClassifier(class_weight="balanced",
                                        random_state=RANDOM_STATE, n_jobs=-1)),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [6, 10, None],
             "clf__min_samples_leaf": [3, 5, 10]},
        ),
    }


def tune(pipeline, grid, X_train, y_train, scoring="f1"):
    """5-fold stratified grid search, optimising F1 on the default class.

    F1 is used instead of accuracy because the classes are imbalanced and we
    care about catching defaulters (recall) without too many false alarms.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, grid, scoring=scoring, cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def evaluate(model, X_test, y_test, threshold=0.5):
    """Return a dict of test-set metrics for the default class (label 1)."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def expected_cost(y_true, pred, fn_cost=5, fp_cost=1):
    """Total misclassification cost.

    The original German Credit documentation states that approving a bad loan
    (false negative) is judged ~5x as costly as rejecting a good customer
    (false positive). We reuse that ratio for the business analysis.
    """
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    fn = ((pred == 0) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    return fn_cost * fn + fp_cost * fp


def feature_names(fitted_pipeline):
    """Feature names after preprocessing, tidied for plotting."""
    names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def group_importance(names, importances, original_cols):
    """Sum one-hot importances back to the original column they came from.

    A categorical variable such as ``savings`` is split into several dummy
    columns; summing them gives a fairer 'whole variable' importance.
    """
    out = {}
    for name, imp in zip(names, importances):
        # longest original column name that prefixes the dummy's name
        match = max((c for c in original_cols if name == c or name.startswith(c + "_")),
                    key=len, default=name)
        out[match] = out.get(match, 0.0) + imp
    return pd.Series(out).sort_values(ascending=False)

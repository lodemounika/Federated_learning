# =============================================================
# STEP 2: CENTRALIZED BASELINE MODEL
# File: baseline/centralized_model.py
# =============================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def train_baseline(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest classifier on the full training set.
    This is the CENTRALIZED baseline — data is not split across nodes.
    """
    print("\n[Baseline] Training Random Forest on full training data...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        random_state=42,
        n_jobs=-1          # use all CPU cores
    )
    model.fit(X_train, y_train)
    print(f"[Baseline] Training complete — {n_estimators} trees")
    return model


def evaluate_model(model, X_test, y_test, label_encoder, model_name="Model"):
    """
    Evaluate a trained model and return a metrics dictionary.
    Prints a full classification report.
    """
    y_pred = model.predict(X_test)

    # ── Per-class metrics ──────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

    # ── ROC-AUC (one-vs-rest) ──────────────────────────────────
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    metrics = {
        'accuracy':  round(acc  * 100, 2),
        'precision': round(prec * 100, 2),
        'recall':    round(rec  * 100, 2),
        'f1_score':  round(f1   * 100, 2),
        'roc_auc':   round(auc  * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  Precision : {metrics['precision']}%")
    print(f"  Recall    : {metrics['recall']}%")
    print(f"  F1-Score  : {metrics['f1_score']}%")
    print(f"  ROC-AUC   : {metrics['roc_auc']}%")
    print(f"{'='*55}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    return metrics


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract top N most important features from the Random Forest.
    Used later to build LLM explanation prompts.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [
        {'feature': feature_names[i], 'importance': round(float(importances[i]), 4)}
        for i in indices
    ]
    return top_features


def save_model(model, path="models/baseline_model.pkl"):
    """Persist the trained model to disk."""
    joblib.dump(model, path)
    print(f"[Baseline] Model saved → {path}")


def load_model(path="models/baseline_model.pkl"):
    """Load a persisted model from disk."""
    return joblib.load(path)

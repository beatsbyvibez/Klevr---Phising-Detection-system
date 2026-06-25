"""
train_model.py
==============
Trains a Random Forest classifier on a labelled URL dataset.

Usage
-----
    python train_model.py --dataset urls.csv

Expected CSV format
-------------------
    url,label
    http://example.com,0
    http://phishing-site.xyz/login,1

Outputs
-------
    model.pkl           – trained classifier
    scaler.pkl          – StandardScaler (optional, kept for pipeline consistency)
    label_encoder.pkl   – LabelEncoder
    model_report.txt    – full evaluation report
    feature_importance.png
    confusion_matrix.png
    roc_curve.png
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

from features import extract_features, feature_names

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset.  Accepts 'url' + 'label' columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "url" in df.columns, "CSV must have a 'url' column"
    assert "label" in df.columns, "CSV must have a 'label' column"
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    print(f"[INFO] Loaded {len(df):,} samples  |  "
          f"Safe={sum(df.label==0):,}  Phishing={sum(df.label==1):,}")
    return df


# ---------------------------------------------------------------------------
# Feature extraction (batch)
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Extracting features…  (this may take a moment)")
    rows = []
    for i, url in enumerate(df["url"]):
        if i % 5000 == 0 and i > 0:
            print(f"         processed {i:,} / {len(df):,}")
        try:
            rows.append(extract_features(str(url)))
        except Exception:
            rows.append({k: 0 for k in feature_names()})
    X = pd.DataFrame(rows, columns=feature_names())
    y = df["label"].values
    print(f"[INFO] Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # Stratified 5-fold cross-validation
    print("[INFO] Running 5-fold stratified cross-validation…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"       CV F1:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    print("[INFO] Fitting final model on full training set…")
    model.fit(X_train, y_train)

    # Predictions
    y_pred     = model.predict(X_test)
    y_proba    = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    print("\n" + "="*55)
    print("  MODEL EVALUATION SUMMARY")
    print("="*55)
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  CV F1 (mean±std)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*55)
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Safe", "Phishing"]))

    return model, X_test, y_test, y_pred, y_proba, {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc,
        "cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Phishing"],
                yticklabels=["Safe", "Phishing"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_roc_curve(y_test, y_proba, auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names       = [feature_cols[i] for i in indices]
    vals        = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2563EB" if v >= np.percentile(vals, 70) else "#93C5FD" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(model, metrics, feature_cols):
    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
    print(f"[INFO] Saved: model.pkl")

    # Save feature column order so inference stays aligned
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_cols.pkl"))

    # Write text report
    report_path = os.path.join(OUTPUT_DIR, "model_report.txt")
    with open(report_path, "w") as f:
        f.write("AI-Powered Phishing URL Detection System\n")
        f.write("Model Evaluation Report\n")
        f.write("=" * 55 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<22}: {v:.4f}\n")
    print(f"[INFO] Saved: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train phishing URL classifier")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    args = parser.parse_args()

    df             = load_dataset(args.dataset)
    X, y           = build_feature_matrix(df)
    model, X_test, y_test, y_pred, y_proba, metrics = train_and_evaluate(X, y)

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, metrics["auc"])
    plot_feature_importance(model, list(X.columns))
    save_artefacts(model, metrics, list(X.columns))

    print("\n[DONE] Training complete. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import pandas as pd

from data.generate_data import generate_credit_data
from src.features import prepare_features, engineer_features, FEATURE_GROUPS
from src.models import split_data, train_models, save_models, cv_scores, get_models, get_smote_pipelines
from src.evaluate import evaluate_all, get_roc_curve_data, feature_importance, find_best_threshold

DATA_PATH = 'data/credit_risk_dataset.csv'
MODELS_PATH = 'models'


def load_or_generate_data(n_samples=10000, force_generate=False):
    if os.path.exists(DATA_PATH) and not force_generate:
        print(f"Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    else:
        print(f"Generating new dataset with {n_samples} samples...")
        df = generate_credit_data(n_samples)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved to {DATA_PATH}")
        return df


def run_pipeline():
    print("=" * 60)
    print("CREDIT SCORING MODEL - PIPELINE")
    print("=" * 60)

    df = load_or_generate_data(10000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Class distribution:\n{df['credit_risk'].value_counts()}")

    print("\n--- Feature Engineering ---")
    df_feat = engineer_features(df)
    print(f"Total features after engineering: {len(df_feat.select_dtypes(include=[np.number]).columns)}")
    for group, feats in FEATURE_GROUPS.items():
        print(f"  {group}: {feats}")

    print("\n--- Preparing Features ---")
    X, y, scaler = prepare_features(df, target_col='credit_risk', scale=True)
    print(f"Feature matrix: {X.shape}")
    print(f"Target: {y.value_counts().to_dict()}")

    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n--- Training Models (with SMOTE) ---")
    models = train_models(X_train, y_train, use_smote=True)
    for name in models:
        cv_mean, cv_std = cv_scores(models[name], X_train, y_train)
        print(f"  {name:25s} CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

    print("\n--- Evaluation ---")
    results_df, predictions, probabilities = evaluate_all(models, X_test, y_test)
    print("\nTest Set Performance:")
    print(results_df.to_string())

    print("\n--- Feature Importance (Top 10) ---")
    for name, model in models.items():
        fi = feature_importance(model, X.columns)
        if fi is not None:
            print(f"\n{name}:")
            print(fi.head(10).to_string(index=False))

    print(f"\n--- Best Thresholds ---")
    for name, model in models.items():
        threshold, best_f1 = find_best_threshold(model, X_test, y_test)
        print(f"  {name:25s} Best Threshold: {threshold:.3f}, F1: {best_f1:.4f}")

    print("\n--- Saving Models ---")
    save_models(models)
    print(f"Models saved to {MODELS_PATH}/")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return models, X_test, y_test, results_df


if __name__ == '__main__':
    run_pipeline()

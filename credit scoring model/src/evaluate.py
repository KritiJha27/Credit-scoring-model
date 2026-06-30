import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)


def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['False Positive Rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return metrics, y_pred, y_proba


def evaluate_all(models, X_test, y_test):
    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        predictions[name] = y_pred
        probabilities[name] = y_proba

    return pd.DataFrame(results).set_index('Model'), predictions, probabilities


def get_confusion_matrix_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm


def get_roc_curve_data(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    return fpr, tpr, auc


def get_precision_recall_curve_data(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    return precision, recall


def _get_inner_model(model):
    if hasattr(model, 'named_steps'):
        return model.named_steps['classifier']
    return model


def feature_importance(model, feature_names):
    inner = _get_inner_model(model)

    if hasattr(inner, 'feature_importances_'):
        importances = inner.feature_importances_
    elif hasattr(inner, 'coef_'):
        importances = np.abs(inner.coef_[0])
    else:
        return None

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)


def find_best_threshold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold, f1_scores[best_idx]

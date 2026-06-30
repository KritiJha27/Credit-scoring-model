import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os


def split_data(X, y, test_size=0.25, random_state=42, stratify=True):
    stratify_param = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=stratify_param
    )


def get_models(random_state=42):
    return {
        'Logistic Regression': LogisticRegression(
            C=0.5, max_iter=2000, random_state=random_state, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            random_state=random_state, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, random_state=random_state,
            class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10,
            random_state=random_state
        ),
    }


def get_smote_pipelines(random_state=42):
    models = get_models(random_state)
    smote = SMOTE(random_state=random_state, sampling_strategy='auto')
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = ImbPipeline([
            ('smote', smote),
            ('classifier', model)
        ])
    return pipelines


def train_models(X_train, y_train, models=None, use_smote=False, random_state=42):
    if models is None:
        models = get_smote_pipelines(random_state) if use_smote else get_models(random_state)

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def cv_scores(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean(), scores.std()


def save_models(models, path='models'):
    os.makedirs(path, exist_ok=True)
    for name, model in models.items():
        filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join(path, filename))


def load_models(path='models'):
    import glob
    models = {}
    for fpath in glob.glob(os.path.join(path, '*.joblib')):
        name = os.path.splitext(os.path.basename(fpath))[0].replace('_', ' ').title()
        models[name] = joblib.load(fpath)
    return models

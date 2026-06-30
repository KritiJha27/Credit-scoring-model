import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.generate_data import generate_credit_data
from src.features import prepare_features, engineer_features, FEATURE_GROUPS
from src.models import split_data, train_models, save_models, cv_scores
from src.evaluate import (
    evaluate_all, get_confusion_matrix_data, get_roc_curve_data,
    get_precision_recall_curve_data, feature_importance, find_best_threshold
)

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("darkgrid")


def load_css():
    st.markdown("""
    <style>
        .main > div { padding: 0rem 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .metric-card {
            background: #1a1a2e;
            border: 1px solid #16213e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .metric-card:hover { transform: translateY(-2px); }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d4ff;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #a0a0b0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-bad {
            font-size: 2rem;
            font-weight: 700;
            color: #ff6b6b;
        }
        .metric-good {
            font-size: 2rem;
            font-weight: 700;
            color: #51cf66;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #e0e0e0;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 8px;
            margin-bottom: 20px;
        }
        .info-text {
            color: #b0b0b0;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data(n_samples=10000):
    df = generate_credit_data(n_samples)
    return df


@st.cache_resource
def train_all_models(X_train, y_train):
    return train_models(X_train, y_train, use_smote=True)


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=11)
    plt.tight_layout()
    return fig


def plot_roc_comparison(results_df, roc_data):
    fig = go.Figure()
    colors = ['#00d4ff', '#51cf66', '#ff6b6b', '#ffd43b', '#cc5de8']

    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        if name in results_df.index:
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'{name} (AUC={auc:.3f})',
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash', width=1.5)
    ))

    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        legend=dict(x=0.6, y=0.1),
        width=700, height=500,
        hovermode='x unified'
    )
    return fig


def plot_pr_comparison(pr_data):
    fig = go.Figure()
    colors = ['#00d4ff', '#51cf66', '#ff6b6b', '#ffd43b', '#cc5de8']

    for i, (name, (precision, recall)) in enumerate(pr_data.items()):
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))

    fig.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark',
        legend=dict(x=0.6, y=0.1),
        width=700, height=500,
        hovermode='x unified'
    )
    return fig


def plot_feature_importance(fi_df, model_name, top_n=20):
    top_features = fi_df.head(top_n)
    colors = ['#00d4ff' if v > 0 else '#ff6b6b' for v in top_features['importance']]

    fig = go.Figure(go.Bar(
        x=top_features['importance'][::-1],
        y=top_features['feature'][::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=np.round(top_features['importance'][::-1], 3),
        textposition='outside'
    ))

    fig.update_layout(
        title=f'Feature Importance - {model_name}',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_dark',
        height=500,
        margin=dict(l=10, r=40, t=40, b=10)
    )
    return fig


def plot_feature_groups(df_feat):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Income vs Debt', 'Credit History Impact',
            'Payment History Distribution', 'Loan Profile',
            'Default Ratio Distribution', 'Financial Stress Index'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'box'}],
               [{'type': 'histogram'}, {'type': 'box'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )

    fig.add_trace(go.Scatter(
        x=df_feat['income'][:1000], y=df_feat['total_debt'][:1000],
        mode='markers', marker=dict(color='#00d4ff', size=4, opacity=0.6),
        name='Income vs Debt'
    ), row=1, col=1)

    good = df_feat[df_feat['credit_risk'] == 'Good']
    bad = df_feat[df_feat['credit_risk'] == 'Bad']

    fig.add_trace(go.Box(y=good['credit_history_length'], name='Good', marker_color='#51cf66'), row=1, col=2)
    fig.add_trace(go.Box(y=bad['credit_history_length'], name='Bad', marker_color='#ff6b6b'), row=1, col=2)

    fig.add_trace(go.Histogram(x=good['late_payments'], name='Good', marker_color='#51cf66', opacity=0.7), row=2, col=1)
    fig.add_trace(go.Histogram(x=bad['late_payments'], name='Bad', marker_color='#ff6b6b', opacity=0.7), row=2, col=1)

    fig.add_trace(go.Box(y=good['loan_amount'], name='Good', marker_color='#51cf66'), row=2, col=2)
    fig.add_trace(go.Box(y=bad['loan_amount'], name='Bad', marker_color='#ff6b6b'), row=2, col=2)

    fig.add_trace(go.Histogram(x=good['default_ratio'], name='Good', marker_color='#51cf66', opacity=0.7), row=3, col=1)
    fig.add_trace(go.Histogram(x=bad['default_ratio'], name='Bad', marker_color='#ff6b6b', opacity=0.7), row=3, col=1)

    fig.add_trace(go.Histogram(x=good['financial_stress_index'], name='Good', marker_color='#51cf66', opacity=0.7), row=3, col=2)
    fig.add_trace(go.Histogram(x=bad['financial_stress_index'], name='Bad', marker_color='#ff6b6b', opacity=0.7), row=3, col=2)

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        title_text="Feature Analysis by Risk Group"
    )
    return fig


def display_metrics_row(metrics_dict):
    cols = st.columns(len(metrics_dict))
    for i, (label, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            if label in ['False Positive Rate', 'Default Rate']:
                val_color = 'metric-bad' if value > 0.3 else 'metric-good'
            elif label in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
                val_color = 'metric-good' if value > 0.75 else ('metric-bad' if value < 0.5 else 'metric-value')
            else:
                val_color = 'metric-value'

            st.markdown(f"""
            <div class="metric-card">
                <div class="{val_color}">{value:.3f}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def risk_prediction_ui(models, X_test, y_test, feature_names):
    st.markdown('<div class="section-header"> Individual Credit Risk Assessment</div>',
                unsafe_allow_html=True)

    with st.form("prediction_form"):
        cols = st.columns(3)
        with cols[0]:
            income = st.number_input('Annual Income ($)', 15000, 250000, 55000, step=5000)
            age = st.slider('Age', 18, 75, 35)
            loan_amount = st.number_input('Loan Amount ($)', 1000, 100000, 15000, step=1000)
            loan_term = st.selectbox('Loan Term (months)', [12, 24, 36, 48, 60, 72, 84], index=3)
        with cols[1]:
            total_debt = st.number_input('Total Existing Debt ($)', 0, 200000, 8000, step=1000)
            late_payments = st.number_input('Late Payments (past 2 yrs)', 0, 30, 1, step=1)
            credit_history = st.slider('Credit History Length (years)', 1, 30, 5)
            num_accounts = st.slider('Credit Accounts', 0, 20, 5)
        with cols[2]:
            employment_len = st.slider('Employment Length (years)', 0, 40, 3)
            dependents = st.selectbox('Dependents', [0, 1, 2, 3, 4], index=0)
            home_own = st.selectbox('Home Ownership', ['Rent', 'Mortgage', 'Own', 'Other'], index=0)
            education = st.selectbox('Education', ['High School', 'Associate', 'Bachelor', 'Master', 'PhD'], index=2)

        submitted = st.form_submit_button(" Assess Credit Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            'age': age, 'income': income, 'loan_amount': loan_amount,
            'loan_term': loan_term, 'credit_history_length': credit_history,
            'num_credit_accounts': num_accounts, 'late_payments': late_payments,
            'num_defaults': 0, 'total_debt': total_debt,
            'employment_length': employment_len, 'home_ownership': home_own,
            'education_level': education, 'marital_status': 'Single',
            'dependents': dependents, 'has_credit_card': 1,
            'credit_card_utilization': 0.3, 'annual_interest_rate': 8.5,
            'prior_loan_defaults': 0,
        }])

        X_input, _, _ = prepare_features(input_data, scale=True)
        X_input = X_input.reindex(columns=feature_names, fill_value=0)

        st.markdown("---")
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

        model_cols = st.columns(len(models))
        for idx, (name, model) in enumerate(models.items()):
            with model_cols[idx]:
                proba = model.predict_proba(X_input)[0, 1]
                pred = 'Bad' if proba >= 0.5 else 'Good'
                risk_pct = proba * 100

                border_color = '#ff6b6b' if pred == 'Bad' else '#51cf66'
                icon = '' if pred == 'Bad' else ''
                st.markdown(f"""
                <div style="background:#1a1a2e; border:2px solid {border_color};
                            border-radius:12px; padding:20px; text-align:center;">
                    <div style="font-size:1.1rem; color:#a0a0b0; margin-bottom:8px;">{name}</div>
                    <div style="font-size:2rem; font-weight:700; color:{border_color};">
                        {icon} {pred}
                    </div>
                    <div style="font-size:1.5rem; color:#00d4ff; margin-top:8px;">
                        {risk_pct:.1f}% Risk
                    </div>
                    <div style="margin-top:12px;">
                        <div style="background:#16213e; border-radius:10px; height:10px; overflow:hidden;">
                            <div style="background:linear-gradient(90deg, #51cf66, #ffd43b, #ff6b6b);
                                        width:{risk_pct}%; height:100%; border-radius:10px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def main():
    load_css()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px 0;">
            <div style="font-size:2.5rem;"> </div>
            <div style="font-size:1.3rem; font-weight:700; color:#00d4ff;">CREDIT SCORING</div>
            <div style="font-size:0.8rem; color:#a0a0b0; letter-spacing:2px;">MACHINE LEARNING MODEL</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        n_samples = st.slider('Sample Size', 1000, 20000, 10000, step=1000,
                              help='Number of records to generate')
        test_size = st.slider('Test Split', 0.1, 0.4, 0.25, 0.05,
                              help='Fraction of data for testing')

        st.markdown("---")
        run_btn = st.button(" Run Full Pipeline", use_container_width=True,
                            type="primary")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#606060;">
            <strong>Models:</strong><br>
            • Logistic Regression<br>
            • Decision Tree<br>
            • Random Forest<br>
            • Gradient Boosting<br><br>
            <strong>Metrics:</strong><br>
            Precision • Recall • F1-Score • ROC-AUC
        </div>
        """, unsafe_allow_html=True)

    st.title(" Credit Scoring Model")
    st.markdown("""
    <div class="info-text">
        Predicting creditworthiness using classification algorithms with engineered financial features.
        Adjust parameters in the sidebar and run the pipeline.
    </div>
    """, unsafe_allow_html=True)

    if 'pipeline_run' not in st.session_state:
        st.session_state.pipeline_run = False

    if run_btn or st.session_state.pipeline_run:
        st.session_state.pipeline_run = True

        with st.spinner("Generating data and training models..."):
            df = load_data(n_samples)

            df_feat = engineer_features(df)
            X, y, scaler = prepare_features(df, target_col='credit_risk', scale=True)
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

            models = train_all_models(X_train, y_train)

            results_df, predictions, probabilities = evaluate_all(models, X_test, y_test)

            roc_data = {}
            pr_data = {}
            for name, model in models.items():
                fpr, tpr, auc = get_roc_curve_data(model, X_test, y_test)
                roc_data[name] = (fpr, tpr, auc)
                precision, recall = get_precision_recall_curve_data(model, X_test, y_test)
                pr_data[name] = (precision, recall)

            fi_data = {}
            for name, model in models.items():
                fi = feature_importance(model, X.columns)
                if fi is not None:
                    fi_data[name] = fi

        tabs = st.tabs([
            "  Data Overview",
            "  Feature Engineering",
            "  Model Training",
            "  Evaluation",
            "  Predict"
        ])

        with tabs[0]:
            st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

            cols = st.columns(4)
            total = len(df)
            good_count = (df['credit_risk'] == 'Good').sum()
            bad_count = (df['credit_risk'] == 'Bad').sum()
            default_rate = bad_count / total

            with cols[0]:
                st.metric("Total Records", f"{total:,}")
            with cols[1]:
                st.metric("Good Credit", f"{good_count:,}", f"{good_count/total*100:.1f}%")
            with cols[2]:
                st.metric("Bad Credit", f"{bad_count:,}", f"{bad_count/total*100:.1f}%")
            with cols[3]:
                st.metric("Default Rate", f"{default_rate:.2%}",
                          delta=None,
                          delta_color="inverse")

            st.markdown("---")
            st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
            fig_eda = plot_feature_groups(df_feat)
            st.plotly_chart(fig_eda, use_container_width=True)

            with st.expander(" Raw Data Sample"):
                st.dataframe(df.head(20), use_container_width=True)

            with st.expander(" Descriptive Statistics"):
                st.dataframe(df.describe(), use_container_width=True)

        with tabs[1]:
            st.markdown('<div class="section-header">Feature Engineering Pipeline</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#1a1a2e; border-radius:10px; padding:20px; margin-bottom:20px;">
                <h4 style="color:#00d4ff; margin:0 0 10px 0;">Engineered Features</h4>
                <p style="color:#b0b0b0; margin:0;">
                    The following derived features are computed from raw financial data to improve model predictive power.
                </p>
            </div>
            """, unsafe_allow_html=True)

            eng_cols = st.columns(2)
            group_descriptions = {
                'Income & Debt': 'debt_to_income_ratio, financial_stress_index, loan_to_income_ratio',
                'Payment History': 'payment_history_score, default_ratio (captures payment reliability)',
                'Credit Profile': 'credit_account_age, credit_utilization_score, has_credit_card',
                'Demographics': 'employment_stability, dependents_per_income, is_high_education, is_homeowner',
            }

            for i, (group, desc) in enumerate(group_descriptions.items()):
                with eng_cols[i % 2]:
                    st.markdown(f"""
                    <div style="background:#16213e; border-left:3px solid #00d4ff; border-radius:5px; padding:15px; margin-bottom:15px;">
                        <div style="font-weight:600; color:#e0e0e0;">{group}</div>
                        <div style="font-size:0.85rem; color:#808090; margin-top:5px;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Feature Distributions by Risk Class</div>', unsafe_allow_html=True)

            feat_select = st.selectbox("Select Feature", list(FEATURE_GROUPS.keys()))
            if feat_select:
                feat_cols = FEATURE_GROUPS[feat_select]
                num_feats = len(feat_cols)
                plot_cols = min(3, num_feats)
                rows = (num_feats + plot_cols - 1) // plot_cols

                fig = make_subplots(rows=rows, cols=plot_cols,
                                    subplot_titles=feat_cols)
                for idx, col_name in enumerate(feat_cols):
                    r = idx // plot_cols + 1
                    c = idx % plot_cols + 1
                    if col_name in df_feat.columns:
                        good_vals = df_feat[df_feat['credit_risk'] == 'Good'][col_name]
                        bad_vals = df_feat[df_feat['credit_risk'] == 'Bad'][col_name]
                        fig.add_trace(go.Histogram(x=good_vals, name='Good',
                                                   marker_color='#51cf66', opacity=0.6,
                                                   legendgroup='Good'), row=r, col=c)
                        fig.add_trace(go.Histogram(x=bad_vals, name='Bad',
                                                   marker_color='#ff6b6b', opacity=0.6,
                                                   legendgroup='Bad'), row=r, col=c)

                fig.update_layout(
                    height=250 * rows,
                    template='plotly_dark',
                    barmode='overlay',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.markdown('<div class="section-header">Model Training Results</div>', unsafe_allow_html=True)

            with st.spinner("Cross-validating models..."):
                cv_results = []
                for name, model in models.items():
                    mean, std = cv_scores(model, X_train, y_train)
                    cv_results.append({'Model': name, 'CV Mean ROC-AUC': f"{mean:.4f}",
                                       'CV Std': f"{std:.4f}"})
                cv_df = pd.DataFrame(cv_results)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="section-header">Cross-Validation Scores</div>', unsafe_allow_html=True)
                st.dataframe(cv_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:#1a1a2e; border-radius:10px; padding:20px;">
                    <table style="width:100%; color:#b0b0b0;">
                        <tr><td>Train Size</td><td style="text-align:right; color:#00d4ff;">{len(X_train):,}</td></tr>
                        <tr><td>Test Size</td><td style="text-align:right; color:#00d4ff;">{len(X_test):,}</td></tr>
                        <tr><td>Features</td><td style="text-align:right; color:#00d4ff;">{X.shape[1]}</td></tr>
                        <tr><td>Models</td><td style="text-align:right; color:#00d4ff;">{len(models)}</td></tr>
                        <tr><td>Cross-Validation</td><td style="text-align:right; color:#00d4ff;">5-Fold</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            fi_tabs = st.tabs(list(fi_data.keys()))
            for idx, (name, _) in enumerate(fi_data.items()):
                with fi_tabs[idx]:
                    if name in fi_data:
                        st.plotly_chart(
                            plot_feature_importance(fi_data[name], name),
                            use_container_width=True
                        )

        with tabs[3]:
            st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

            display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity', 'False Positive Rate']
            for model_name in results_df.index:
                st.markdown(f"##### {model_name}")
                metrics_dict = {}
                for col in display_cols:
                    if col in results_df.columns:
                        metrics_dict[col] = results_df.loc[model_name, col]

                display_metrics_row(metrics_dict)
                st.markdown("---")

            st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                fig_roc = plot_roc_comparison(results_df, roc_data)
                st.plotly_chart(fig_roc, use_container_width=True)
            with col2:
                fig_pr = plot_pr_comparison(pr_data)
                st.plotly_chart(fig_pr, use_container_width=True)

            st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
            cm_tabs = st.tabs(list(models.keys()))
            for idx, (name, model) in enumerate(models.items()):
                with cm_tabs[idx]:
                    cm = get_confusion_matrix_data(model, X_test, y_test)
                    fig = plot_confusion_matrix(cm, name)
                    st.pyplot(fig)

                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    <div style="display:flex; gap:20px; justify-content:center; margin-top:10px;">
                        <div style="background:#16213e; padding:10px 20px; border-radius:8px; text-align:center;">
                            <div style="color:#51cf66; font-weight:700;">{tn}</div>
                            <div style="color:#a0a0b0; font-size:0.8rem;">True Negatives</div>
                        </div>
                        <div style="background:#16213e; padding:10px 20px; border-radius:8px; text-align:center;">
                            <div style="color:#ff6b6b; font-weight:700;">{fp}</div>
                            <div style="color:#a0a0b0; font-size:0.8rem;">False Positives</div>
                        </div>
                        <div style="background:#16213e; padding:10px 20px; border-radius:8px; text-align:center;">
                            <div style="color:#ff6b6b; font-weight:700;">{fn}</div>
                            <div style="color:#a0a0b0; font-size:0.8rem;">False Negatives</div>
                        </div>
                        <div style="background:#16213e; padding:10px 20px; border-radius:8px; text-align:center;">
                            <div style="color:#51cf66; font-weight:700;">{tp}</div>
                            <div style="color:#a0a0b0; font-size:0.8rem;">True Positives</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Detailed Classification Report</div>', unsafe_allow_html=True)
            st.dataframe(results_df.style.highlight_max(axis=0, color='#00d4ff40'), use_container_width=True)

        with tabs[4]:
            risk_prediction_ui(models, X_test, y_test, X.columns)

    else:
        st.info("Click **Run Full Pipeline** in the sidebar to start.")

        st.markdown("""
        <div style="display:flex; gap:20px; margin-top:30px; flex-wrap:wrap;">
            <div style="background:#1a1a2e; border-radius:12px; padding:25px; flex:1; min-width:200px; border-top:3px solid #00d4ff;">
                <div style="font-size:1.2rem; font-weight:600; color:#00d4ff;">Data Generation</div>
                <div style="color:#a0a0b0; font-size:0.9rem; margin-top:8px;">Synthetic dataset with 20+ financial features including income, debt, payment history, and demographics.</div>
            </div>
            <div style="background:#1a1a2e; border-radius:12px; padding:25px; flex:1; min-width:200px; border-top:3px solid #51cf66;">
                <div style="font-size:1.2rem; font-weight:600; color:#51cf66;">Feature Engineering</div>
                <div style="color:#a0a0b0; font-size:0.9rem; margin-top:8px;">Debt-to-income ratio, payment history score, credit utilization, financial stress index, and more.</div>
            </div>
            <div style="background:#1a1a2e; border-radius:12px; padding:25px; flex:1; min-width:200px; border-top:3px solid #ffd43b;">
                <div style="font-size:1.2rem; font-weight:600; color:#ffd43b;">Model Training</div>
                <div style="color:#a0a0b0; font-size:0.9rem; margin-top:8px;">Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting with 5-fold CV.</div>
            </div>
            <div style="background:#1a1a2e; border-radius:12px; padding:25px; flex:1; min-width:200px; border-top:3px solid #ff6b6b;">
                <div style="font-size:1.2rem; font-weight:600; color:#ff6b6b;">Evaluation</div>
                <div style="color:#a0a0b0; font-size:0.9rem; margin-top:8px;">Precision, Recall, F1-Score, ROC-AUC, confusion matrices, and interactive charts.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()

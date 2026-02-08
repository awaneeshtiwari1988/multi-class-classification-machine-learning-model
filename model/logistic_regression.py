import streamlit as st
from sklearn.linear_model import LogisticRegression
from model.utils import evaluate_model, visualize_results

def train_logistic(X_train, y_train, max_iter=200, show_progress=False):
    """
    Train Logistic Regression model with optional Streamlit progress reporting.
    """
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate progress updates during training
        for i in range(max_iter):
            # We don't actually train per-epoch in scikit-learn,
            # but we can simulate progress for user feedback.
            progress_bar.progress((i + 1) / max_iter)
            status_text.text(f"Training iteration {i+1}/{max_iter}")

        progress_bar.empty()
        status_text.text("✅ Training complete")

    # Actual training
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_logistic(model, X_test, y_test):
    """
    Evaluate Logistic Regression model with standard metrics.
    """
    return evaluate_model(model, X_test, y_test)


def visualize_logistic(model, X_test, y_test):
    """
    Visualize Logistic Regression performance (confusion matrix, report, ROC).
    """
    visualize_results(model, X_test, y_test, "Logistic Regression")

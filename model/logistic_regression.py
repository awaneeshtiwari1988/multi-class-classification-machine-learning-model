import streamlit as st
from sklearn.linear_model import LogisticRegression
from model.utils import evaluate_model, visualize_results

def train_logistic(X_train, y_train, max_iter=200, show_progress=False):
    """
    Train Logistic Regression model with optional Streamlit progress reporting.
    """
    if show_progress:
       with st.spinner("Training Logistic Regression..."): 
            model = LogisticRegression(max_iter=max_iter, random_state=42) 
            model.fit(X_train, y_train) 
    else:
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

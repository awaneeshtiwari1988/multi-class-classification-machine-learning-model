import streamlit as st
import pandas as pd

from model import logistic_regression
from model.data_loader import load_data, preprocess_data
from model.utils import evaluate_model, visualize_results

st.title("🌲 Multi-Class Classification with ML Models")
st.write("Choose to upload your own dataset or run directly on the inbuilt **forest_cover_data**.")
st.write("Run models on the inbuilt **forest_cover_data** dataset.")

# Sidebar: model selection
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Execution button
run_button = st.button("Run Model")

if run_button:
    # -------------------------------
    # Data loading
    # -------------------------------
    X, y = load_data("data/covtype.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)

    # -------------------------------
    # Train model with progress
    # -------------------------------
    st.subheader("Training Progress")

    if model_choice == "Logistic Regression":
        model = logistic_regression.train_logistic(X_train_scaled, y_train, show_progress=True)
    elif model_choice == "Decision Tree":
        model = decision_tree.train_decision_tree(X_train_scaled, y_train)
    elif model_choice == "KNN":
        model = knn.train_knn(X_train_scaled, y_train)
    elif model_choice == "Naive Bayes":
        model = naive_bayes.train_naive_bayes(X_train_scaled, y_train)
    elif model_choice == "Random Forest":
        model = random_forest.train_random_forest(X_train_scaled, y_train)
    elif model_choice == "XGBoost":
        model = xgboost.train_xgboost(X_train_scaled, y_train)

    # -------------------------------
    # Evaluate + Visualize
    # -------------------------------
    st.subheader("Evaluation Metrics")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    st.write(pd.DataFrame(metrics, index=["Score"]).T)

    st.subheader("Visualizations")
    visualize_results(model, X_test_scaled, y_test, model_choice)
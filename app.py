import streamlit as st
import pandas as pd

from model import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost
from model.data_loader import load_data, preprocess_data, load_uploaded_data
import model.utils as utils

st.title("🌲 Multi-Class Classification with ML Models")
st.write("Choose to upload your own dataset or run directly on the inbuilt **forest_cover_data**.")

# Sidebar: model selection
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File uploader
uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])

# Execution button
run_button = st.button("Run Model")

if run_button:
    # -------------------------------
    # Data loading logic
    # -------------------------------
    if uploaded_file is not None:
        st.success("Using uploaded file for training and evaluation.")
        # Train on default dataset 
        X, y = load_data(uploaded_file)
        X_train_scaled, X_test_scaled_default, y_train, y_test_default, scaler = preprocess_data(X, y)

        # Load uploaded test data and scale with same scaler 
        X_test_uploaded, y_test_uploaded = utils.load_uploaded_test_data(uploaded_file)
        X_test_scaled = scaler.transform(X_test_uploaded)
        y_test = y_test_uploaded
    else:
        st.info("No file uploaded. Running model on the inbuilt dataset: **forest_cover_data**")
        X, y = load_data("covtype.csv")
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)

    # -------------------------------
    # Train model
    # -------------------------------
    progress_placeholder = st.empty() 
    progress_placeholder.subheader("Training Progress")
    model = utils.train_model(model_choice, X_train_scaled, y_train)
    progress_placeholder.subheader("✅ Training completed")

    # -------------------------------
    # Evaluate + Visualize
    # -------------------------------
    eval_placeholder = st.empty() 
    eval_placeholder.subheader("Evaluation started...")
    # For models like XGBoost that require 0-based labels, pass label_offset=1 
    if model_choice == "XGBoost": 
        metrics = utils.evaluate_model(model, X_test_scaled, y_test, label_offset=1) 
    else: 
        metrics = utils.evaluate_model(model, X_test_scaled, y_test, label_offset=0)

    eval_placeholder.subheader("✅ Evaluation completed")

    st.subheader("Evaluation Metrics")
    st.write(pd.DataFrame(metrics, index=["Score"]).T)

    st.subheader("Visualizations")
    utils.visualize_results(model, X_test_scaled, y_test, model_choice)
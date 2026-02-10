import streamlit as st
import pandas as pd

from model import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost
from model.data_loader import load_data, preprocess_data
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
        X, y = load_data(uploaded_file)
    else:
        st.info("No file uploaded. Running model on the inbuilt dataset: **forest_cover_data**")
        X, y = load_data("data/covtype.csv")

    # Preprocess
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X, y)

    # -------------------------------
    # Train model
    # -------------------------------
    progress_placeholder = st.empty() 
    progress_placeholder.subheader("Training Progress")
    if model_choice == "Logistic Regression":
        model = logistic_regression.train_logistic(X_train_scaled, y_train, max_iter=200, batch_fraction=0.2)
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
    # Once training completes, update the placeholder 
    progress_placeholder.subheader("✅ Training completed")

    # -------------------------------
    # Evaluate + Visualize
    # -------------------------------
    eval_placeholder = st.empty() 
    eval_placeholder.subheader("Evaluation started...")
    metrics = utils.evaluate_model(model, X_test_scaled, y_test, y_train=y_train)
    eval_placeholder.subheader("✅ Evaluation completed")

    st.subheader("Evaluation Metrics")
    st.write(pd.DataFrame(metrics, index=["Score"]).T)

    st.subheader("Visualizations")
    utils.visualize_results(model, X_test_scaled, y_test, model_choice)
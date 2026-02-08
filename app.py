import streamlit as st
import pandas as pd
from model import logistic_regression
#from model import decision_tree
#from model import knn
#from model import naive_bayes
#from model import random_forest
#from model import xgboost

from model.data_loader import load_data, preprocess_data
from model.utils import evaluate_model, visualize_results


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌲 Multi-Class Classification with ML Models")
st.write("Repository: **multi-class-classification-machine-learning-model**")
st.write("Upload a CSV file, select a model, and evaluate performance.")

# Sidebar: model selection
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File uploader
uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])

if uploaded_file is not None:
    # Load uploaded data
    X, y = load_data(uploaded_file)
    # Preprocess (scale features)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)

    # Train model based on choice
    if model_choice == "Logistic Regression":
        model = logistic_regression.train_logistic(X_train_scaled, y_train)
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

    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    st.subheader("Evaluation Metrics")
    st.write(pd.DataFrame(metrics, index=["Score"]).T)

    # Visualize
    st.subheader("Visualizations")
    visualize_results(model, X_test_scaled, y_test, model_choice)

else:
    st.info("Please upload a CSV file to evaluate.")
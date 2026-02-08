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
st.write("Choose to upload your own dataset or run directly on the inbuilt **Forest Cover dataset**.")

# Sidebar: model selection
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File uploader
uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])

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
X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)

# -------------------------------
# Train model
# -------------------------------
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

# -------------------------------
# Evaluate + Visualize
# -------------------------------
metrics = evaluate_model(model, X_test_scaled, y_test)
st.subheader("Evaluation Metrics")
st.write(pd.DataFrame(metrics, index=["Score"]).T)

st.subheader("Visualizations")
visualize_results(model, X_test_scaled, y_test, model_choice)
import streamlit as st
import pandas as pd
import joblib
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

if "startup_done" not in st.session_state:
    # Run only once at app startup
    X, y = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_path = preprocess_data(X, y, save_test=True)

    st.session_state.startup_done = True
    st.session_state.scaler = scaler
    st.session_state.X_test_scaled = X_test_scaled
    st.session_state.y_test = y_test
    st.session_state.test_path = test_path

# Provide download link for test.csv 
with open(st.session_state.test_path, "rb") as f: 
    st.download_button(label="Download standardized test.csv", data=f, file_name="test.csv", mime="text/csv" )


# -----------------------------
# Helper functions
# -----------------------------
def load_model(model_choice):
    """Load pretrained model from .pkl files."""
    filename_map = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "KNN": "models/knn.pkl",
        "Naive Bayes": "models/naive_bayes.pkl",
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl"
    }
    return joblib.load(filename_map[model_choice])

use_pretrained = st.sidebar.checkbox("Use Pretrained Models (.pkl)", value=True)

# Execution button
run_button = st.button("Run Model")

if run_button:
    # -------------------------------
    # Data loading logic
    # -------------------------------
    # Initialize variables so they exist in scope 
    X_train_scaled, y_train = None, None 
    X_test_scaled, y_test = None, None 
    scaler = None
    if uploaded_file is not None:
        scaler = joblib.load("models/scaler.pkl")
        X_test_uploaded, y_test_uploaded = load_uploaded_data(uploaded_file)
        X_test_scaled = scaler.transform(X_test_uploaded)
        y_test = y_test_uploaded
    else:
        st.info("No file uploaded. Running model on the inbuilt dataset: **forest_cover_data**")
        X, y = load_data("covtype.csv")
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_path = preprocess_data(X, y)

    # -------------------------------
    # Train model
    # -------------------------------
    progress_placeholder = st.empty() 
    progress_placeholder.subheader("Training Progress")
    if use_pretrained:
        model = load_model(model_choice)
    else:
        if X_train_scaled is not None and y_train is not None: 
            model = utils.train_model(model_choice, X_train_scaled, y_train) 
        else: 
            st.warning("Fresh training is not possible with uploaded test data. Using pretrained model instead.") 
            model = load_model(model_choice)
        # Train fresh if needed
        model = utils.train_model(model_choice, X_train_scaled, y_train)
    
    progress_placeholder.subheader("✅ Training completed")

    # -------------------------------
    # Evaluate + Visualize
    # -------------------------------
    eval_placeholder = st.empty() 
    eval_placeholder.subheader("Evaluation started...")
    metrics = utils.evaluate_model(model, X_test_scaled, y_test, label_offset=1 if model_choice == "XGBoost" else 0)
    eval_placeholder.subheader("✅ Evaluation completed")

    st.subheader("Evaluation Metrics")
    st.write(pd.DataFrame(metrics, index=["Score"]).T)

    st.subheader("Visualizations")
    utils.visualize_results(model, X_test_scaled, y_test, model_choice)
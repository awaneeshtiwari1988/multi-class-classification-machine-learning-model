import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os

def load_uploaded_data(uploaded_file):
    """
    Load the uploaded standardized test.csv file and return X, y.
    This file already has headers and the Cover_Type column.
    """
    # Read uploaded file (with headers, since you saved test.csv with header=True)
    data = pd.read_csv(uploaded_file)

    # Separate features and target
    X = data.drop("Cover_Type", axis=1)
    y = data["Cover_Type"].astype(int)

    return X, y

def load_data(file_name = "covtype.csv"):
    # Ensure data folder exists 
    data_dir = "data" 
    os.makedirs(data_dir, exist_ok=True) 
    file_path = os.path.join(data_dir, file_name) 

    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz" 
    
    progress_placeholder = st.empty()

    # Check if file exists locally 
    if os.path.exists(file_path): 
        progress_placeholder.subheader("Loading dataset from local file...")
        data = pd.read_csv(file_path, header=None) 
        progress_placeholder.subheader("✅ Dataset load complete (local file).")
    else: 
        progress_placeholder.subheader("Downloading dataset from UCI repository...") 
        data = pd.read_csv(uci_url, header=None, compression="gzip") 
        data.to_csv(file_path, index=False, header=False)
        progress_placeholder.subheader("✅ Dataset download and load complete (UCI).")

    # Column names from UCI documentation
    columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ] + [f"Wilderness_Area_{i}" for i in range(4)] \
    + [f"Soil_Type_{i}" for i in range(40)] \
    + ["Cover_Type"]

    # Assign column names
    data.columns = columns
    
    X = data.drop("Cover_Type", axis=1)
    y = data["Cover_Type"]
    return X, y

def preprocess_data(X, y, save_test=False):
    """
    Split and scale the dataset.
    Optionally save test split as test.csv.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    test_path = None
    if save_test:
        test_df = pd.DataFrame(X_test, columns=X.columns)
        test_df["Cover_Type"] = y_test.values
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        test_path = os.path.join(data_dir, "test.csv")
        test_df.to_csv(test_path, index=False, header=True)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_path

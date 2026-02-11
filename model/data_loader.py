import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os

def load_uploaded_data(uploaded_file):
    """
    Load an uploaded CSV file and return X, y in the same format
    as the default covtype dataset.
    """
    # Read uploaded file into DataFrame
    data = pd.read_csv(uploaded_file, header=None)

    # Apply the same column names as the default dataset
    columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ] + [f"Wilderness_Area_{i}" for i in range(4)] \
      + [f"Soil_Type_{i}" for i in range(40)] \
      + ["Cover_Type"]

    data.columns = columns
    X = data.drop("Cover_Type", axis=1)
    y = data["Cover_Type"]
    return X, y

def load_data(file_name = "covtype.csv"):
    # Ensure data folder exists 
    data_dir = "data" 
    os.makedirs(data_dir, exist_ok=True) 
    file_path = os.path.join(data_dir, file_name) 

    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz" 
    
    # Check if file exists locally 
    if os.path.exists(file_path): 
        st.write("Loading dataset from local file...") 
        data = pd.read_csv(file_path, header=None) 
    else: 
        st.write("Downloading dataset from UCI repository...") 
        data = pd.read_csv(uci_url, header=None, compression="gzip") 
        data.to_csv(file_path, index=False, header=False)

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


def preprocess_data(X, y):
    """
    Split and scale the dataset.
    
    Parameters
    ----------
    X : DataFrame
        Feature matrix
    y : Series
        Target labels
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed
    
    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test
    """

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

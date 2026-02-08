import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="data/covtype.csv"):
    # Column names from UCI documentation
    columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ] + [f"Wilderness_Area_{i}" for i in range(4)] \
      + [f"Soil_Type_{i}" for i in range(40)] \
      + ["Cover_Type"]

    # Load dataset without header, then assign column names
    df = pd.read_csv(path, header=None)
    df.columns = columns

    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
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
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_and_preprocess_data(path="data/covtype.csv", test_size=0.2, random_state=42):
    """
    Convenience wrapper to load and preprocess data in one step.
    """
    X, y = load_data(path)
    return preprocess_data(X, y, test_size=test_size, random_state=random_state)

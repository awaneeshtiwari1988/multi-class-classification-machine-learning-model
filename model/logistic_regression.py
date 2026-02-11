import streamlit as st
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.utils import shuffle

def train_logistic(X_train, y_train, max_iter=1000, batch_fraction=0.2):
    """
    Train Logistic Regression model with optional Streamlit progress reporting.
    """
    # Initialize model with saga solver and warm_start
    log_reg = LogisticRegression(
        max_iter=1, solver='saga', random_state=42, warm_start=True
    )
    
    # Iterative training with progress bar
    n_samples = int(len(X_train) * batch_fraction)
    for i in tqdm(range(max_iter), desc="Training Logistic Regression"):
        X_batch, y_batch = shuffle(X_train, y_train, random_state=i)
        log_reg.fit(X_batch[:n_samples], y_batch[:n_samples])
    
    return log_reg

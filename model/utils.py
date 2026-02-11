from operator import le
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from model import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost
from sklearn.preprocessing import label_binarize
import streamlit as st

def train_model(model_choice, X_train, y_train):
    """
    Train a model based on the user's choice.
    Returns the trained model.
    """
    if model_choice == "Logistic Regression":
        return logistic_regression.train_logistic(X_train, y_train, max_iter=200, batch_fraction=0.2)
    elif model_choice == "Decision Tree":
        return decision_tree.train_decision_tree(X_train, y_train)
    elif model_choice == "KNN":
        return knn.train_knn(X_train, y_train)
    elif model_choice == "Naive Bayes":
        return naive_bayes.train_naive_bayes(X_train, y_train)
    elif model_choice == "Random Forest":
        return random_forest.train_random_forest(X_train, y_train)
    elif model_choice == "XGBoost":
        return xgboost.train_xgboost(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

def evaluate_model(model, X_test, y_test, label_offset=1):
    # Shift labels for AUC calculation 
    y_test_adj = y_test - label_offset
    # Predictions 
    y_pred_adj = model.predict(X_test) 
    y_proba = model.predict_proba(X_test)

    # Restore predictions to original label space 
    y_pred = y_pred_adj + label_offset

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    y_proba = model.predict_proba(X_test)
    metrics["AUC"] = roc_auc_score(y_test_adj, y_proba, multi_class='ovr')

    return metrics

def visualize_results(model, X_test, y_test, title="Model"):
    """
    Visualize model performance: confusion matrix, classification report heatmap, ROC curves.
    
    Parameters
    ----------
    model : fitted classifier
    X_test : ndarray
        Test features (scaled)
    y_test : Series
        Test labels
    title : str
        Title for plots
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    #plt.show()
    st.pyplot(plt)

    # Classification Report Heatmap
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10,6))
    sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
    plt.title(f"Classification Report Heatmap - {title}")
    #plt.show()
    st.pyplot(plt)

    # ROC Curve (multi-class)
    y_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(10,8))
    for i in range(len(classes)): 
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i]) 
        plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"ROC Curve - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    #plt.show()
    st.pyplot(plt)

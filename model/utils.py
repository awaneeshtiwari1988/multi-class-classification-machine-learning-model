import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model with standard metrics.
    
    Parameters
    ----------
    model : fitted classifier
    X_test : ndarray
        Test features (scaled)
    y_test : Series
        Test labels
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Adjust labels if they start from 1 (Forest Cover dataset)
    y_adj = y_test - 1 if y_test.min() == 1 else y_test

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": roc_auc_score(y_adj, y_proba, multi_class="ovr")
    }

    # Try AUC only if class counts match 
    try: 
        metrics["AUC"] = roc_auc_score(y_test, y_proba, multi_class="ovr") 
    except ValueError:
        metrics["AUC"] = None

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

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Classification Report Heatmap
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10,6))
    sns.heatmap(df.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
    plt.title(f"Classification Report Heatmap - {title}")
    plt.show()

    # ROC Curve (multi-class)
    y_adj = y_test - 1 if y_test.min() == 1 else y_test
    y_bin = label_binarize(y_adj, classes=list(range(len(set(y_adj)))))
    plt.figure(figsize=(10,8))
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i+1} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"ROC Curve - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

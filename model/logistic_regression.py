from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from tqdm import tqdm
from model.data_loader import load_and_preprocess_data
from model.utils import evaluate_model, visualize_results

def train_logistic(X_train=None, y_train=None, max_iter=200, batch_fraction=0.2):
    """
    Train Logistic Regression with iterative batches and progress bar.
    
    Parameters
    ----------
    X_train : ndarray
        Training features (scaled)
    y_train : Series
        Training labels
    max_iter : int
        Number of iterations (default=200)
    batch_fraction : float
        Fraction of training data used per iteration (default=0.2)
    
    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model
    """
    # If no data passed, load default dataset
    if X_train is None or y_train is None:
        X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

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


def evaluate_logistic(model, X_test, y_test):
    """
    Evaluate Logistic Regression model with standard metrics.
    """
    return evaluate_model(model, X_test, y_test)


def visualize_logistic(model, X_test, y_test):
    """
    Visualize Logistic Regression performance (confusion matrix, report, ROC).
    """
    visualize_results(model, X_test, y_test, "Logistic Regression")

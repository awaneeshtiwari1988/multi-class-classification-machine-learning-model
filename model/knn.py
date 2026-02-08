from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, n_neighbors=5, weights="uniform", metric="minkowski"):
    """
    Train a K-Nearest Neighbors classifier.
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_neighbors : int
        Number of neighbors to use
    weights : str
        Weight function ("uniform" or "distance")
    
    Returns
    -------
    model : KNeighborsClassifier
    """
    knn_clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    
    knn_clf.fit(X_train, y_train)
    
    return knn_clf

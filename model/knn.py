from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, n_neighbors=5, weights="uniform"):
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
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    return knn

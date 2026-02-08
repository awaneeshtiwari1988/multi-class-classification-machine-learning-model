from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, criterion="gini", max_depth=None):
    """
    Train a Decision Tree classifier.
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    max_depth : int, optional
        Maximum depth of the tree
    random_state : int
        Random seed
    
    Returns
    -------
    model : DecisionTreeClassifier
    """
    dt_clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )
    
    dt_clf.fit(X_train, y_train)
    
    return dt_clf
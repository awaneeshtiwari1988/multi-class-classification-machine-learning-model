from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
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
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt
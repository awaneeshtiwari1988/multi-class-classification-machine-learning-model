from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest classifier.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of each tree
    random_state : int
        Random seed
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
    """
    Train an XGBoost classifier.
    
    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum depth of trees
    learning_rate : float
        Step size shrinkage
    random_state : int
        Random seed
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="mlogloss"
    )
    y_train_adj = y_train - 1
    xgb.fit(X_train, y_train_adj)
    return xgb

from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,colsample_bytree=0.8,
                  random_state=42):
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
    y_train_adj = y_train - 1 
    xgb_clf = XGBClassifier(n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            max_depth=max_depth, 
                            subsample=subsample, 
                            colsample_bytree=colsample_bytree, 
                            random_state=random_state, 
                            use_label_encoder=False, 
                            eval_metric="mlogloss" )
    xgb_clf.fit(X_train, y_train_adj)
    return xgb_clf
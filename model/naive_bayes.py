from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    """
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

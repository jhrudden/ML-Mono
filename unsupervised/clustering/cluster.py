class BaseClusterMixin(object):
    """
    Base class for all clustering algorithms

    Attributes:
        n_clusters: int, number of clusters
        max_iter: int, maximum number of iterations
        tol: float, tolerance to declare convergence
        random_state: int, random seed
    
    Methods:
        fit(X): Fit the model to the data
        predict(X): Predict the cluster of each sample
        fit_predict(X): Fit the model and predict the cluster of each sample
    """
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        raise NotImplementedError("Fit method not implemented")

    def predict(self, X):
        raise NotImplementedError("Predict method not implemented")

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
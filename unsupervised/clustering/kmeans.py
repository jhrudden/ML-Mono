import numpy as np

class KMeans:
    """
    My implementation of KMeans clustering algorithm
    
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form as well as the number of centroids to generate.
    
    Attributes
    ----------
    labels_ : array, shape (n_samples,)
        Labels of each point
    centroids_ : array, shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.
    """
    def __init__(self, n_clusters: int = 2):
        assert n_clusters > 0
        self.n_clusters = n_clusters
        self._labels = None
        self._centroids = None
    
    def fit(self, X, num_iterations: int = 200, tol: float = 1e-4):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(num_iterations):
            labels = self.predict(X)
            new_centroids = [X[labels == i].mean(axis=0) for i in range(self.n_clusters)]

            if np.allclose(self.centroids, new_centroids, atol=tol):
                break

            self.centroids = new_centroids
            self._labels = labels
        
    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)
        

        

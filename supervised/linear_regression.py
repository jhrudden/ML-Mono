import numpy as np

class LinearRegression:
    _weights = None
    def __init__(self):
        self._weights = None

    def fit_cfs(self, X, y):
        """
        Fit the linear regression model using closed form solution
        """
        assert X.shape[0] == y.shape[0]

        X = np.hstack((np.ones((X.shape[0], 1)), X)) # add bias term
        self._weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # closed form solution
    
    def fit_gd(self, X, y, lr=0.01, epochs=1000):
        """
        Fit the linear regression model using gradient descent
        """
        assert X.shape[0] == y.shape[0]

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self._weights = np.random.randn(X.shape[1])
        for _ in range(epochs):
            preds = np.dot(X, self._weights)
            dw = np.dot(X.T, (preds - y)) / X.shape[0]
            if np.allclose(dw, 0, atol=1e-5) or np.any(np.isnan(dw)): # early stopping for negligible gradients
                break
            self._weights -= lr * dw
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self._weights)
    

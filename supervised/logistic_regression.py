import numpy as np

class LogisticRegression:
    _weights = None
    def __init__(self):
        self.weights = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, num_iterations: int = 1_000, lr=1e-3): # notice no closed form solution as first order derivative is none linear
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.random.randn(X.shape[1])
        for _ in range(num_iterations):
            dw = np.dot(X.T, self._sigmoid(X @ self.weights) - y) / y.shape[0] # first order derivative 
            # (Notice: looks like linear regression)
            self.weights -= lr * dw
        
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self._sigmoid(X @ self.weights)


    

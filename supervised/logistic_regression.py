import numpy as np

class LogisticRegression:
    _weights = None
    def __init__(self):
        self._weights = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y, num_iterations: int = 10_000, lr=1e-1): # notice no closed form solution as first order derivative is none linear
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self._weights = np.random.randn(X.shape[1])
        losses = []
        for _ in range(num_iterations):
            dw = (np.dot(X.T, self._sigmoid(X @ self._weights) - y)) / y.shape[0] # first order derivative
            # (Notice: looks like linear regression)
            self._weights -= lr * dw

            if _ % 100 == 0:
                losses.append(self._loss(y, self._sigmoid(X @ self._weights)))


        return losses
        
    def predict(self, X, as_prob: bool = False):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_hat = self._sigmoid(X @ self._weights).reshape(-1, 1)
        if as_prob:
            return np.hstack((1 - y_hat, y_hat))
        return np.round(y_hat).reshape(-1)
    
    def predict_proba(self, X):
        return self.predict(X, as_prob=True)


    

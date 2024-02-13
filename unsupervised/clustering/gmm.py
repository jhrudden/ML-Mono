# trying EM
from scipy.stats import multivariate_normal
import numpy as np
from .base import BaseClusterMixin

class GaussianMixture(BaseClusterMixin):
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4):
        super().__init__(name="GaussianMixture", n_clusters=n_clusters, max_iter=max_iter, tol=tol)
        self.weight_matrix = None
        self.phis = None
        self.means = None
        self.covs = None

    def _init(self, X):
        """
        Given some data X, initialize the parameters of the model.
        
        Params:
            X: (n_samples, n_features) array
        """
        n_samples, n_features = X.shape
        self.phis = np.ones(self.n_clusters) / self.n_clusters
        self.weight_matrix = np.ones((n_samples, self.n_clusters)) / self.n_clusters
        self.means = np.random.choice(X.flatten(), size=(self.n_clusters, n_features))
        self.covs = np.array([np.cov(X, rowvar=False) for _ in range(self.n_clusters)])

    def _e_step(self, X):
        """
        Perform the E-step of the EM algorithm. 
        This step calculates the responsibilities for each sample. Responsibilities
        represent the probability that each sample belongs to each cluster.

        Params:
            X: (n_samples, n_features) array
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            responsibilities[:, i] = multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i])
        sum_responsibilities = responsibilities.dot(self.phis)
        return responsibilities / sum_responsibilities[:, np.newaxis]

    def _m_step(self, X):
        """
        Perform the M-step of the EM algorithm.
        This step updates the parameters of the model to maximize the likelihood of the data based on the
        responsibilities calculated in the E-step.

        Params:
            X: (n_samples, n_features) array
        """
        for i in range(self.n_clusters):
            weights = self.weight_matrix[:, i]
            total_weight = weights.sum()
            self.means[i] = np.sum(X * weights[:, np.newaxis], axis=0) / total_weight
            centered = X - self.means[i]
            self.covs[i] = np.dot(centered.T, centered * weights[:, np.newaxis]) / total_weight

    def fit(self, X):
        """
        Fit the model to the data. This method runs the EM algorithm to update the parameters
        of the model to maximize the likelihood of the data.

        Params:
            X: (n_samples, n_features) array
        """
        self._init(X)
        prev_log_likelihood = -np.infty
        for _ in range(self.max_iter):
            self.weight_matrix = self._e_step(X)
            self.phis = self.weight_matrix.mean(axis=0)
            self._m_step(X)
            
            # Check for convergence
            log_likelihood = self._log_likelihood(X)
            if log_likelihood - prev_log_likelihood < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def predict(self, X):
        """
        Predict the cluster for each sample in X.
        Params:
            X: (n_samples, n_features) array

        Returns:
            labels: (n_samples,) array of cluster labels
        """
        self.weight_matrix = self._e_step(X)
        return np.argmax(self.weight_matrix, axis=1)
    
    def _log_likelihood(self, X):
        """
        Calculate the log likelihood of the data given the model.
        Params:
            X: (n_samples, n_features) array
        Returns:
            mle: float, log likelihood of the data
        """
        m, _ = X.shape
        probs = np.zeros((m, self.n_clusters))
        for i in range(self.n_clusters):
            probs[:, i] = self.phis[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i])
        
        return np.sum(np.log(probs.sum(axis=1)))
        
    def pdf(self, X):
        """Calculate the probability density function for each sample in X."""
        pdf_values = np.zeros(X.shape[0])
        for i in range(self.n_clusters):
            pdf_values += self.phis[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i])
        return pdf_values
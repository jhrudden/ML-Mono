# trying EM
from scipy.stats import multivariate_normal

class GaussianMixture:
    def __init__(self, num_components=2, max_iter=100):
        self.num_components = num_components
        self.max_iter = max_iter
        self.weight_matrix = None
        self.phis = None
        self.means = None
        self.covs = None

    def _init(self, X):
        n_samples, n_features = X.shape
        self.phis = np.ones(self.num_components) / self.num_components
        self.weight_matrix = np.ones((n_samples, self.num_components)) / self.num_components
        self.means = np.random.choice(X.flatten(), size=(self.num_components, n_features))
        self.covs = np.array([np.cov(X, rowvar=False) for _ in range(self.num_components)])

    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.num_components))
        for i in range(self.num_components):
            responsibilities[:, i] = multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i])
        sum_responsibilities = responsibilities.dot(self.phis)
        return responsibilities / sum_responsibilities[:, np.newaxis]

    def _m_step(self, X):
        for i in range(self.num_components):
            weights = self.weight_matrix[:, i]
            total_weight = weights.sum()
            self.means[i] = np.sum(X * weights[:, np.newaxis], axis=0) / total_weight
            centered = X - self.means[i]
            self.covs[i] = np.dot(centered.T, centered * weights[:, np.newaxis]) / total_weight

    def fit(self, X):
        self._init(X)
        for _ in range(self.max_iter):
            self.weight_matrix = self._e_step(X)
            self.phis = self.weight_matrix.mean(axis=0)
            self._m_step(X)

    def predict(self, X):
        """
        Predict the cluster for each sample in X.
        Params:
            X: (n_samples, n_features) array

        Returns:
            labels: (n_samples,) array of cluster labels
        """
        self.weight_matrix = self.e_step(X)
        return np.argmax(self.weight_matrix, axis=1)
    
    def fit_predict(self, X):
        """
        Fit the model to the data and predict the cluster labels
        Params:
            X: (n_samples, n_features) array
        Returns:
            labels: (n_samples,) array of cluster labels
        """
        self.fit(X)
        return self.predict(X)

    def pdf(self, X):
        """Calculate the probability density function for each sample in X."""
        pdf_values = np.zeros(X.shape[0])
        for i in range(self.num_components):
            pdf_values += self.phis[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i])
        return pdf_values
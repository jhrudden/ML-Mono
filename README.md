# ML-Mono

Collections of rough implementations of various ML concepts for learning purposes.

## Project Structure

This repository is organized into several directories, each focusing on different aspects of machine learning.

### Supervised Learning

Implementations of supervised learning algorithms:

-   **Linear Regression**: See `supervised/linear_regression.py`.
-   **Logistic Regression**: See `supervised/logistic_regression.py`.
-   **Neural Networks**:
    -   Recurrent Neural Networks (RNNs):
        -   Simple RNN: `supervised/neural_networks/rnn/simple_rnn.py`.
        -   LSTM (Long Short-Term Memory): `supervised/neural_networks/rnn/lstm.py`.
        -   RNN-based Language Model: `supervised/neural_networks/rnn/language_model.py`.

### Unsupervised Learning

Implementations of unsupervised learning algorithms:

-   **Clustering**:
    -   K-Means Clustering: `unsupervised/clustering/kmeans.py`.

### Metrics

Currently implemented metrics for model evaluation:

-   **Accuracy**: For evaluating classification models. See `metrics/classification_metrics.py`.

### Notebooks

Jupyter notebooks demonstrating the implementation and usage of various algorithms:

-   **Linear Regression Example**: `notebooks/LinRegression_Example.ipynb`.
-   **Logistic Regression Example**: `notebooks/LogRegression_Example.ipynb`.
-   **Training RNNs to Generate Content Inspired by Shakespeare**: `notebooks/RNN_Shakespeare_Training.ipynb`.

#### TODO:

1. Add mixin for Unsupervised & Supervised Models to implement `fit_predict`, `fit`, `predict` methods + `X`, `y` fields.

## License

This project is licensed under the [MIT License](LICENSE).

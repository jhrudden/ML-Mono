import numpy as np

def accuracy_score(y_true, y_pred) -> float:
    """
    Calculates the accuracy between the true labels and the predicted labels.
    :param y_true: true labels
    :param y_pred: predicted labels

    :return: accuracy score between 0 and 1
    """
    assert len(y_true) == len(y_pred), "The length of the true labels and the predicted labels must be the same."
    return np.mean(y_true == y_pred)
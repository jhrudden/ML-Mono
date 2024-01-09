import tensorflow_datasets as tfds
import torch
import numpy as np

import sys
sys.path.append('..')
from unsupervised.clustering import KMeans

def load_tf_dataset(dataset_name: str, split_name: str = 'train', text_key: str = 'text'):
    """
    Load lines from a specified TensorFlow dataset and split.

    Parameters:
    - dataset_name (str): Name of the dataset to load.
    - split_name (str): Name of the split to load (default is 'train').
    - text_key (str): Key in the dataset that contains the text (default is 'text').

    Returns:
    - str: All lines from the specified dataset and split concatenated into a single string.
    """

    try:
        # Load the specified dataset and split
        dataset = tfds.load(dataset_name, split=split_name, as_supervised=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}' with split '{split_name}': {e}")

    data = []
    for example in dataset:
        try:
            text = example[text_key].numpy().decode('utf-8')
            data.append(text)
        except KeyError:
            raise KeyError(f"Key '{text_key}' not found in the dataset '{dataset_name}'. Please specify the correct key.")

    return data

def expand_sequence_for_rnn_training(data, labels, vocab_size: int, seq_len: int):
    """
    Expands each sequence into a set of incremental subsequences for RNN training.

    :param data: Tensor of input data representing sequences.
    :param labels: Tensor of label data corresponding to the input sequences.
    :param vocab_size: Size of the vocabulary for one-hot encoding.
    :param seq_len: The sequence length to consider for expansion.
    :return: A tuple (contexts, embedded_labels), where contexts is a list of context tensors,
             and embedded_labels is a tensor of one-hot encoded labels.
    """
    embedded_labels = torch.nn.functional.one_hot(labels, num_classes=vocab_size).float()
    contexts = []
    for i in range(seq_len):
        contexts.append(data[:, :i + 1])
    return contexts, embedded_labels

def generate_regression_dataset(num_samples, degree, noise, x_range=(0, 1), seed=42):
    """
    Generate a dataset with a polynomial relationship between a single feature and target. 
    
    Parameters:
        num_samples (int): number of samples to generate
        degree (int): degree of polynomial
        noise (float): standard deviation of gaussian noise
        x_range (tuple): range of x values
        seed (int): random seed
    Returns:
        X (num_samples, 1): feature vector for unscaled x value
        y (np.ndarray): target vector
        degree (int): degree of polynomial
    """
    assert degree > 0, "degree must be greater than 0"
    np.random.seed(seed)

    # generate random weights
    w = np.random.uniform(-1, 1, degree+1)

    X = np.random.uniform(x_range[0], x_range[1], (num_samples, 1))
    
    for i in range(2, degree+1):
        X = np.hstack((X, np.power(X[:,0], i).reshape(-1, 1)))
    
    X = np.hstack((np.ones((num_samples, 1)), X)) # add bias term

    noise = np.random.normal(0, noise, num_samples)

    y = np.dot(X, w) + noise

    return X[:,1].reshape(-1, 1), y, degree

def generate_dataset_for_classification(n_samples, n_features, n_classes, center_box: tuple = (-10.0, 10.0), random_state: int = 42):
    """
    Generate a dataset for classification task by generating points from a Gaussian distribution around a random centeroids. Samples
    are then assigned to the nearest centeroid using KMeans to reduce the variance of each class.

    Parameters:
        n_samples (int): number of samples to generate
        n_features (int): number of features
        n_classes (int): number of classes
        center_box (tuple): range of centeroid values
        random_state (int): random seed
    
    Returns:
        X (np.ndarray): feature matrix
        y (np.ndarray): target vector (each sample's class label)
    """
    np.random.seed(random_state)
    centers = np.random.uniform(low=center_box[0], high=center_box[1], size=(n_classes, n_features))
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, 1))
    for i in range(n_samples):
        center = np.random.choice(n_classes)
        sample = np.random.normal(loc=centers[center], scale=1.0)
        X[i, :] = sample
    
    # TODO implement own version of Kmeans
    kmeans = KMeans(n_clusters=n_classes)
    y = kmeans.fit_predict(X)
    return X, y
    
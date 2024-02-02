import tensorflow_datasets as tfds
import torch
import numpy as np
from typing import Tuple, Union, List

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

def make_circles_2d(n_samples: int, noise: float = 0.1, factor: float = 0.5, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D data points arranged in two concentric circles.

    Parameters:
    n_samples (int): Total number of samples to generate.
    noise (float): Standard deviation of Gaussian noise added to the data.
    factor (float): Scale factor between the sizes of the two circles.
    random_state (int): Seed for the random number generator.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the generated points (X) and their corresponding labels (y).
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if noise < 0 or factor < 0:
        raise ValueError("noise and factor must be non-negative numbers")

    np.random.seed(random_state)

    num_outer_samples = int(n_samples / 2)
    num_inner_samples = n_samples - num_outer_samples

    outer_lin = np.linspace(0, 2 * np.pi, num_outer_samples)
    inner_lin = np.linspace(0, 2 * np.pi, num_inner_samples)

    outer_x = np.cos(outer_lin) + np.random.normal(scale=noise, size=num_outer_samples)
    outer_y = np.sin(outer_lin) + np.random.normal(scale=noise, size=num_outer_samples)

    inner_x = np.cos(inner_lin) * factor + np.random.normal(scale=noise, size=num_inner_samples)
    inner_y = np.sin(inner_lin) * factor + np.random.normal(scale=noise, size=num_inner_samples)

    X = np.vstack((np.hstack((outer_x, inner_x)), np.hstack((outer_y, inner_y)))).T
    y = np.hstack((np.zeros(num_outer_samples), np.ones(num_inner_samples)))

    return X, y

    
def make_moons_2d(n_samples: int, noise: float, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D data points arranged in two interleaving half circles.

    Parameters:
    n_samples (int): Total number of samples to generate.
    noise (float): Standard deviation of Gaussian noise added to the data.
    random_state (int): Seed for the random number generator.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the generated points (X) and their corresponding labels (y).
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if noise < 0:
        raise ValueError("noise must be a non-negative number")

    np.random.seed(random_state)

    num_top_samples = int(n_samples / 2)
    num_bot_samples = n_samples - num_top_samples

    top_circ_x = np.cos(np.linspace(0, np.pi, num_top_samples))
    top_circ_y = np.sin(np.linspace(0, np.pi, num_top_samples))

    bot_circ_x = 1 - np.cos(np.linspace(0, np.pi, num_bot_samples))
    bot_circ_y = 1 - np.sin(np.linspace(0, np.pi, num_bot_samples)) - .5

    X = np.vstack((np.hstack((top_circ_x, bot_circ_x)), np.hstack((top_circ_y, bot_circ_y)))).T
    y = np.hstack((np.zeros(num_top_samples), np.ones(num_bot_samples)))

    if noise > 0:
        X += np.random.normal(scale=noise, size=(n_samples, 2))
    return X, y

def generate_uniform_noise(n_samples, n_features, random_state=42):
    """
    Generate a dataset with uniform noise.

    Parameters:
    n_samples (int): Number of samples to generate.
    n_features (int): Number of features for each sample.
    random_state (int): Seed for the random number generator.

    Returns:
    Tuple[np.ndarray, None]: Dataset with uniform noise. No labels :( cause why would you label noise?
    """
    rng = np.random.RandomState(random_state)
    data = rng.rand(n_samples, n_features)
    return data

def make_blobs(n_samples:int, n_features:int = 2, centers:Union[float, List[List[float]]] = 3, cluster_std:Union[float, List[float]] = 1.0, bounding_box:Tuple[float, float] = (10,-10), random_state:int = 42):
    """
    Generate isotropic Gaussian blobs for clustering.

    Parameters:
    n_samples (int): Total number of samples to generate.
    n_features (int): Number of features for each sample.
    centers (int): Number of clusters to generate.
    cluster_std (float or list of floats): Standard deviation of the clusters.
    bounding_box (tuple): Range of values for the centers of the clusters.
    random_state (int): Seed for the random number generator.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the generated points (X) and their corresponding labels (y).
    """
    n_centers = centers
    if isinstance(centers, int):
        if centers <= 0:
            raise ValueError("centers must be a positive integer")
        centers = np.random.uniform(low=bounding_box[1], high=bounding_box[0], size=(n_centers, n_features))
    elif isinstance(centers, list):
        n_centers = len(centers)
    else:
        raise ValueError("centers must be a positive integer or a list coordinates of the centers of the clusters.")

    if isinstance(cluster_std, float):
        cluster_std = [cluster_std] * n_centers
    elif isinstance(cluster_std, list):
        if len(cluster_std) != n_centers:
            raise ValueError("cluster_std must be a float or a list of floats with length equal to centers")
        n_centers = len(cluster_std)
    else:
        raise ValueError("cluster_std must be a float or a list of floats with length equal to centers")

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    
    if n_samples < n_centers:
        raise ValueError("n_samples must be greater than or equal to centers")
    
    np.random.seed(random_state)

    samples_per_cluster = n_samples // n_centers
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    for i, center in enumerate(centers):
        start = i * samples_per_cluster
        end = (i + 1) * samples_per_cluster
        if i == n_centers - 1:
            end = n_samples
        X[start:end] = np.random.normal(loc=center, scale=cluster_std[i], size=(end-start, n_features))
        y[start:end] = i
    
    return X, y




    

    
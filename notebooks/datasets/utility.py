import tensorflow_datasets as tfds
import torch

def load_dataset(dataset_name: str, split_name: str = 'train', text_key: str = 'text'):
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

    return "\n".join(data)  # Joining with newline character, can be adjusted as needed

def prepare_sequence_contexts(data, labels, vocab_size: int, seq_len: int):
    """
    Prepares context sequences and corresponding embedded labels for training a sequence model.

    :param data: Tensor of input data representing sequences.
    :param labels: Tensor of label data corresponding to the input sequences.
    :param vocab_size: Size of the vocabulary for one-hot encoding.
    :param seq_len: The sequence length to consider for context building.
    :return: A tuple (contexts, embedded_labels), where contexts is a list of context tensors,
             and embedded_labels is a tensor of one-hot encoded labels.
    """
    embedded_labels = torch.nn.functional.one_hot(labels, num_classes=vocab_size).float()
    contexts = []
    for i in range(seq_len):
        contexts.append(data[:, :i + 1])
    return contexts, embedded_labels

def num_learnable_params(model: torch.nn.Module):
    """
    Calculates the number of learnable parameters in a model.

    :param model: The model to calculate the number of learnable parameters for.
    :return: The number of learnable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

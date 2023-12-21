import tensorflow_datasets as tfds

def load_dataset(dataset_name: str, split_name: str = 'train'):
    """
    Load lines from a specified TensorFlow dataset and split.

    Parameters:
    - dataset_name (str): Name of the dataset to load.
    - split_name (str): Name of the split to load (default is 'train').

    Returns:
    - List[str]: A list of lines extracted from the dataset.
    """

    # Load the specified dataset and split
    dataset = tfds.load(dataset_name, split=split_name)

    # Extract lines
    lines = []
    for example in dataset:
        text = example['text'].numpy().decode('utf-8')
        current_lines = text.split('\n')
        current_lines = list(filter(lambda line: len(line) > 0, current_lines))
        lines.extend(current_lines)

    return lines

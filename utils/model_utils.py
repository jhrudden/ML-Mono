import torch

def num_learnable_params(model: torch.nn.Module):
    """
    Calculates the number of learnable parameters in a model.

    :param model: The model to calculate the number of learnable parameters for.
    :return: The number of learnable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

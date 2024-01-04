import torch
import torch.nn as nn

class BaseRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BaseRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def _check_dimensions_forward(self, X):
        assert X.dim() == 3 and X.size(2) == self.input_dim, f'Invalid input dimensions. Expected 3D tensor with dimensions (batch_size, seq_len, input_dim), got {X.size()}'
    
    # TODO: maybe this shouldn't do both X and h checks
    def _check_dimensions_step(self, X, h):
        assert X.dim() == 2, f'X dim: {X.dim()}'
        assert h.dim() == 2, f'h_n_minus_1 dim: {h.dim()}'
        assert X.size(0) == h.size(0), f'X size: {X.size()}, h size: {h.size()}'
        assert X.size(1) == self.input_dim, f'X size: {X.size()}, input_dim: {self.input_dim}'
        assert h.size(1) == self.hidden_dim, f'h size: {h.size()}, hidden_dim: {self.hidden_dim}'

    def _initialize_hidden_state(self, X, h_0):
        if h_0 is None:
            return torch.zeros((X.size(0), self.hidden_dim), device=X.device)
        assert h_0.dim() == 2, f'h dim: {h_0.dim()}'
        assert X.size(0) == h_0.size(0), f'X size: {X.size()}, h size: {h_0.size()}'
        assert h_0.size(1) == self.hidden_dim, f'h size: {h_0.size()}, hidden_dim: {self.hidden_dim}'
        return h_0

    def forward(self, X, h_0=None):
        raise NotImplementedError("This method should be overridden by subclasses")
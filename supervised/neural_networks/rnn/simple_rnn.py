# Simple Recurrent Neural Network (Elman Network) Implementation in PyTorch
# Reference (conceptual):
# Jurafsky, D., & Martin, J.H. (2022). "RNNs and LSTMs." In Speech and Language Processing (3rd ed.). 
# Retrieved from https://web.stanford.edu/~jurafsky/slp3/

import torch
from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W = nn.Parameter(torch.zeros(hidden_dim, input_dim))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W)

    
    def forward(self, X, h_0: torch.Tensor = None):
        """
        Forward pass of the model.
        :param X: input tensor of shape (batch_size, seq_len, input_dim)
        :param h: hidden state tensor of shape (batch_size, hidden_dim)
        :return: output tensor of shape (batch_size, seq_len, hidden_dim)
                 final hidden state tensor of shape (batch_size, hidden_dim)
        """
        # double check X and h dimensions
        assert X.dim() == 3, f'X dim: {X.dim()}'
        assert X.size(2) == self.input_dim, f'X size: {X.size()}, input_dim: {self.input_dim}'

        if h_0 is None:
            h_0 = torch.zeros((X.size(0), self.hidden_dim))
        else:
            assert h_0.dim() == 2, f'h dim: {h_0.dim()}'
            assert X.size(0) == h_0.size(0), f'X size: {X.size()}, h size: {h_0.size()}'
            assert h_0.size(1) == self.hidden_dim, f'h size: {h_0.size()}, hidden_dim: {self.hidden_dim}'
        
        out_tensor = torch.zeros((X.size(0), X.size(1), self.hidden_dim))
        h = h_0
        for t in range(X.size(1)):
            x_t = X[:, t, :]
            h = self.step(x_t, h) # get new hidden state
            out_tensor[:, t, :] = h
        return out_tensor, h


    def step(self, X, h_n_minus_1):
        """
        Single step in time of the model.
        :param X: input tensor of shape (batch_size, input_dim)
        :param h_n_minus_1: hidden state tensor of shape (batch_size, hidden_dim)
        :return: output tensor of shape (batch_size, hidden_dim)
        """
        # double check X and h dimensions
        assert X.dim() == 2, f'X dim: {X.dim()}'
        assert h_n_minus_1.dim() == 2, f'h_n_minus_1 dim: {h_n_minus_1.dim()}'
        assert X.size(0) == h_n_minus_1.size(0), f'X size: {X.size()}, h_n_minus_1 size: {h_n_minus_1.size()}'
        assert X.size(1) == self.input_dim, f'X size: {X.size()}, input_dim: {self.input_dim}'
        assert h_n_minus_1.size(1) == self.hidden_dim, f'h_n_minus_1 size: {h_n_minus_1.size()}, hidden_dim: {self.hidden_dim}'

        # calculate contribution from previous hidden state
        hidden_contribution = torch.matmul(self.U, h_n_minus_1.T)
        input_contribution = torch.matmul(self.W, X.T)
 
        hidden_input = hidden_contribution + input_contribution

        # calculate new hidden state
        h_n = torch.tanh(hidden_input) # (hidden_dim, batch_size)

        return h_n.T # (batch_size, hidden_dim)

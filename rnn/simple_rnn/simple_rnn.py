# Simple Recurrent Neural Network Implementation
# Reference (conceptual)):
# Jurafsky, D., & Martin, J.H. (2022). "Recurrent Neural Networks." In Speech and Language Processing (3rd ed.). 
# Retrieved from https://web.stanford.edu/~jurafsky/slp3/9.pdf

from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
        self.W = nn.Parameter(torch.rand(hidden_dim, input_dim))
        self.V = nn.Parameter(torch.rand(output_dim, hidden_dim))

        self.hidden_state = torch.zeros((hidden_dim, 1))

        # mental model of hidden layer products
        # input_dim = n
        # hidden_dim = m
        # U @ h_t-1 = m x m @ m x 1 = m x 1
        # W @ x = m x n @ n x 1 = m x 1
        # => m x 1 + m x 1 = m x 1
        # so hidden state output is m x 1 (same as hidden_dim) check!
    
    def step(self, x):
        # calculate contribution from previous hidden state
        hidden_contribution = self.U @ self.hidden_state
        input_contribution = self.W @ x
        hidden_input = hidden_contribution + input_contribution

        # calculate new hidden state
        self.hidden_state = torch.tanh(hidden_input)

        # calculate output
        logits = self.V @ self.hidden_state
        return logits

    def get_initial_prev_activation(self):
        return torch.zeros((self.hidden_dim, 1))
        
# Simple Recurrent Neural Network Implementation
# Reference (conceptual)):
# Jurafsky, D., & Martin, J.H. (2022). "Recurrent Neural Networks." In Speech and Language Processing (3rd ed.). 
# Retrieved from https://web.stanford.edu/~jurafsky/slp3/9.pdf

import torch
from torch import nn

# TODO: have SimpleRNN take a foward function that takes input: (batch_size, seq_len, input_dim) and hidden layer: (batch_size, hidden_dim) and returns output: (batch_size, seq_len, hidden_dim)
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

    
    def forward(self, X, h_0):
        """
        Forward pass of the model.
        :param X: input tensor of shape (batch_size, seq_len, input_dim)
        :param h: hidden state tensor of shape (batch_size, hidden_dim)
        :return: output tensor of shape (batch_size, seq_len, hidden_dim)
                 final hidden state tensor of shape (batch_size, hidden_dim)
        """
        # double check X and h dimensions
        assert X.dim() == 3, f'X dim: {X.dim()}'
        assert h_0.dim() == 2, f'h dim: {h_0.dim()}'
        assert X.size(0) == h_0.size(0), f'X size: {X.size()}, h size: {h_0.size()}'
        assert X.size(2) == self.input_dim, f'X size: {X.size()}, input_dim: {self.input_dim}'
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


class LanguageModelSRNN(nn.Module):
    """
    Language model implemented with a simple RNN.
    """
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, use_embedding: bool = False, embedding_dim: int = None):
        super(LanguageModelSRNN, self).__init__()
        self.rnn = SimpleRNN(vocab_size, hidden_dim, vocab_size)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim if use_embedding else vocab_size

        if use_embedding:
            assert embedding_dim is not None, f'embedding_dim must be specified if use_embedding is True'
            self.embedding_dim = embedding_dim
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
    def embedding(self, X):
        """
        Returns the embedding of the input. If use_embedding is False, returns a one hot encoding of the input.

        :param X: input tensor of shape (batch_size, seq_len)
        :return: embedding tensor of shape (batch_size, seq_len, embedding_dim)
        """
        if self.use_embedding:
            return self.embedding_layer(X)
        else:
            return torch.nn.functional.one_hot(X, num_classes=self.embedding_dim).float()
    
    def forward(self, X):
        """
        Forward pass of the model.
        :param x: input tensor of shape (batch_size, seq_len)
        :return: output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # one hot encode input
        X_embedding = self.embedding(X) # (batch_size, seq_len, vocab_size)
        B, S, V = X_embedding.size()
        outputs = torch.zeros((B, S, V))
        h_0 = torch.zeros((B, self.rnn.hidden_dim))
        logits, _ = self.rnn(X_embedding, h_0)

        # A little sad to have to re loop through the sequence
        for t in range(S):
            h_t = logits[:, t, :]
            logits_t = self.linear(h_t)
            softmaxed = torch.softmax(logits_t, dim=1)
            outputs[:, t, :] = softmaxed

        return outputs
    
    @torch.no_grad()
    def generate(self, context, seq_len: int, num_samples: int = 1):
        """
        Generate a sequence of text using the model.
        :param seq_len: the length of the sequence to generate
        :return: a sequence of text
        """
        self.eval()
        h = torch.zeros((1, self.rnn.hidden_dim))
        for i in range(num_samples):
            current_context = context[-min(seq_len, len(context)):]
            current_context = self.embedding(current_context.view(1, -1))
            _, h = self.rnn(current_context, h)
            logits = self.linear(h)
            softmaxed = torch.softmax(logits, dim=1)
            next_char = torch.multinomial(softmaxed, num_samples=1).view(-1)
            context = torch.cat([context, next_char], dim=0)
        resultant_sequence = context.detach().tolist()
        self.train()
        return resultant_sequence

        
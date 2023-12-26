# Simple Recurrent Neural Network Implementation
# Reference (conceptual)):
# Jurafsky, D., & Martin, J.H. (2022). "Recurrent Neural Networks." In Speech and Language Processing (3rd ed.). 
# Retrieved from https://web.stanford.edu/~jurafsky/slp3/9.pdf

import torch
from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.V = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.reset_hidden_state()


    
    def step(self, X):
        if X.dim == 1:
            X = X.view(1, -1)
        print(f'hidden state size: ', self.hidden_state.size())
        # calculate contribution from previous hidden state
        hidden_contribution = self.U @ self.hidden_state
        input_contribution = self.W @ X.T
        hidden_input = hidden_contribution + input_contribution
        print(f'sizes of hidden_contribution, input_contribution, hidden_input: ', hidden_contribution.size(), input_contribution.size(), hidden_input.size())

        # calculate new hidden state
        self.hidden_state = torch.tanh(hidden_input)

        # calculate output
        logits = self.V @ self.hidden_state
        return logits.T # transpose to match batch size

    def reset_hidden_state(self):
        self.hidden_state = torch.zeros((self.hidden_dim, 1))

class LanguageModelSRNN(nn.Module):
    """
    Language model implemented with a simple RNN.
    """
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int):
        super(LanguageModelSRNN, self).__init__()
        self.rnn = SimpleRNN(vocab_size, hidden_dim, vocab_size)
    
    def forward(self, x):
        """
        Forward pass of the model.
        :param x: input tensor of shape (batch_size, seq_len)
        :return: output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # one hot encode input
        x = self.embedding(x)
        outputs = []
        self.rnn.reset_hidden_state()
        for i in range(x.shape[1]):
            logits = self.rnn.step(embedding[:, i, :])
            softmaxed = torch.softmax(logits, dim=1)
            outputs.append(softmaxed)
        
        return torch.stack(outputs, dim=1)
    
    @torch.no_grad()
    def generate(self, context, seq_len: int, num_samples: int = 1):
        """
        Generate a sequence of text using the model.
        :param seq_len: the length of the sequence to generate
        :return: a sequence of text
        """
        self.eval()
        self.rnn.reset_hidden_state()
        for i in range(num_samples):
            current_context = context[:min(seq_len, len(context))]
            current_context = self.embedding(current_context.view(1, -1))
            for char in current_context[0]:
                logits = self.rnn.step(char.T)
            softmaxed = torch.softmax(logits, dim=1)
            next_char = torch.multinomial(softmaxed, num_samples=1).view(-1)
            context = torch.cat([context, next_char], dim=0)
        self.train()
        return context.T.detach().tolist()
    
    def embedding(self, x):
        """
        Embedding layer for the model.
        :param x: input tensor of shape (batch_size, seq_len)
        :return: output tensor of shape (batch_size, seq_len, vocab_size)
        """
        return torch.nn.functional.one_hot(x, num_classes=self.rnn.input_dim).float()
        